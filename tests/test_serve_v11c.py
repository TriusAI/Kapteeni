"""v1.1c serving tests (torch-free; MockV11C + the real wire code paths):
the state.image extension, E1-style independence through the v1.1c
interfaces, contract errors on bad images, and the HTTP server end to
end (API + the demo's static assets)."""

import base64
import copy
import io
import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from kapteeni.contract import ContractError, validate_request
from kapteeni.mock import MockV11C
from kapteeni.model_v11c import extract_image, question_passes
from kapteeni.serve_v11c import make_handler
from kapteeni.serialize import state_text


def tiny_png() -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (200, 30, 30)).save(buf, "PNG")
    return buf.getvalue()


def img_state(note="a small red square", b64=None) -> dict:
    return {"image": b64 or base64.b64encode(tiny_png()).decode(),
            "note": note}


# ----------------------------------------------------------- extract_image

def test_extract_image_placeholder_and_passthrough():
    st = img_state(note="keep me")
    pass_state, img = extract_image(st)
    assert pass_state["image"] == "<attached>"
    assert pass_state["note"] == "keep me"
    assert img is not None and img.size == (8, 8)
    # no image -> unchanged; string/array states pass through untouched
    assert extract_image({"note": "x"})[1] is None
    assert extract_image("plain state")[1] is None
    assert extract_image([1, 2])[0] == [1, 2]


def test_extract_image_data_uri():
    b64 = base64.b64encode(tiny_png()).decode()
    pass_state, img = extract_image(
        {"image": f"data:image/png;base64,{b64}"})
    assert img is not None and pass_state["image"] == "<attached>"


def test_extract_image_errors():
    for bad in (None, 123, "", "not-base64!!", "AQIDBA=="):  # last: not an image
        with pytest.raises(ContractError):
            extract_image({"image": bad})
    assert extract_image({"other": "field"})[1] is None  # no image key


def test_extract_image_content_keyed_not_spelling_keyed():
    """The mock keys on image CONTENT: two spellings of the same bytes
    (bare base64 vs data URI) must give identical answers."""
    m = MockV11C()
    b64 = base64.b64encode(tiny_png()).decode()
    q = {"q": {"type": "noul", "instructions": "is it red?",
               "criteria": {"true": "red", "false": "not red"}}}
    a1, _ = m.evaluate({"image": b64, "note": "x"}, q)
    a2, _ = m.evaluate({"image": f"data:image/png;base64,{b64}", "note": "x"}, q)
    assert a1 == a2


# ------------------------------------------------------------- pass layout

def test_question_passes_match_training_layout():
    st = {"image": "<attached>", "note": "a cafe menu."}
    noul = question_passes({"type": "noul",
                            "instructions": "Is it open?",
                            "criteria": {"true": "open", "false": "shut"}},
                           st)
    assert len(noul) == 1
    assert noul[0]["text"].startswith('{"image": "<attached>", "note"')
    assert "[noul] Is it open?" in noul[0]["text"]
    ch = question_passes({"type": "choice", "instructions": "pick",
                          "criteria": {"A": None, "B": "the second"}},
                         st)
    assert len(ch) == 2
    assert "option - A" in ch[0]["text"] and "option - B: the second" in ch[1]["text"]
    sc = question_passes({"type": "score", "instructions": "rate",
                          "criteria": ["low", "high"]}, st)
    assert "level 1 of 2 - low" in sc[0]["text"]


def test_vision_tokens_matches_measured_grids():
    """Anchored to the 2026-10-02 processor probe (grid_thw values)."""
    from kapteeni.model_v11c import vision_tokens
    assert vision_tokens(640, 640) == 400      # the training figure
    assert vision_tokens(1024, 768) == 768
    assert vision_tokens(1920, 1080) == 2040
    assert vision_tokens(4000, 3000) == 11750
    # above the 16.78 MP processor budget: proportional estimate
    assert abs(vision_tokens(5000, 4000) - 16300) < 300


def test_usage_counts_real_image_tokens():
    """The serving path standardizes images to 640x640, so the served
    (standardized) model always bills the same ~400-token image cost;
    the mock counts by the raw dims it was given (no standardization on
    the mock path)."""
    m = MockV11C()
    q = {"q": {"type": "noul", "instructions": "x", "criteria": None}}
    _, u1 = m.evaluate(img_state(), q)
    _, u2 = m.evaluate(img_state(b64=_b64_of(1600, 1200)), q)
    assert u2["input_tokens"] > u1["input_tokens"]


def test_bound_image_policy():
    """Serving image policy: downscale to a 640px longest edge only when
    larger; smaller images pass UNCHANGED (no upscale, no crop, no
    paste — resizing/pasting measurably flipped the church-hours real
    photo that the raw pixels answered correctly)."""
    from PIL import Image
    from kapteeni.model_v11c import bound_image
    big = Image.new("RGB", (1600, 800), (10, 200, 30))
    out = bound_image(big)
    assert out.size == (640, 320)  # longest edge bounded, aspect intact
    small = Image.new("RGB", (300, 200), (10, 200, 30))
    assert bound_image(small).size == (300, 200)   # untouched
    assert bound_image(Image.new("RGBA", (800, 400),
                                 (1, 2, 3, 255))).size == (640, 320)


def _b64_of(w, h):
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (10, 200, 30)).save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def test_server_notice_on_oversized_image(server):
    """Any image the bounding actually changes (>640px longest edge)
    gets a shrinkage/accuracy notice; images at or under the bound
    (even photos above 0.5 MP generally) get none."""
    for w, h, want in ((1600, 1200, True), (800, 800, True),
                       (640, 640, False), (8, 8, False)):
        body = {"state": {"image": _b64_of(w, h), "note": "photo"},
                "model": "kapteeni-v1.1c",
                "questions": {"q": {"type": "noul", "instructions": "x",
                                    "criteria": None}}}
        status, resp = post(server, body)
        assert status == 200
        if want:
            assert "notice" in resp and "downscaled to a 640px" \
                in resp["notice"]
            assert "accuracy may differ" in resp["notice"]
        else:
            assert "notice" not in resp
    body = {"state": img_state(),
            "model": "kapteeni-v1.1c",
            "questions": {"q": {"type": "noul", "instructions": "x",
                                "criteria": None}}}
    status, resp = post(server, body)
    assert status == 200 and "notice" not in resp


def test_warmup_noop_on_mock():
    """The mock's warmup is a no-op — the interface contract carries it."""
    m = MockV11C()
    m.warmup([(640, 640), (1600, 900)])  # must not raise


# --------------------------------------------------------------------- E1

def test_e1_independence_with_images():
    """Adding/removing questions never changes another question's answer;
    ids never reach the model (same content, different ids -> identical)."""
    m = MockV11C()
    st = img_state()
    q1 = {"type": "noul", "instructions": "is it red?",
          "criteria": {"true": "red", "false": "not"}}
    q2 = {"type": "choice", "instructions": "which?",
          "criteria": {"A": None, "B": None}}
    a1, _ = m.evaluate(st, {"x": q1})
    a12, _ = m.evaluate(st, {"x": q1, "y": q2})
    assert a12["x"] == a1["x"]
    a_copy, _ = m.evaluate(st, {"renamed": q1})
    assert a_copy["renamed"] == a1["x"]
    # repeated calls are stable (determinism)
    assert m.evaluate(st, {"x": q1})[0] == a1


# ----------------------------------------------------------------- server

@pytest.fixture(scope="module")
def server():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0),
                               make_handler(MockV11C(), None, "kapteeni-v1.1c"))
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def post(url, body):
    req = urllib.request.Request(
        url + "/v1/systemone", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_server_models_and_demo(server):
    with urllib.request.urlopen(server + "/v1/models") as r:
        body = json.loads(r.read())
    names = [m["name"] for m in body["models"]]
    assert "kapteeni-v1.1c" in names and "jev-latest" in names
    with urllib.request.urlopen(server + "/") as r:
        html = r.read().decode()
    assert "kapteeni" in html and "Workbench" in html
    with urllib.request.urlopen(server + "/demo/cases.json") as r:
        cases = json.loads(r.read())
    assert cases["cases"] and cases["model"] == "kapteeni-v1.1c"
    for c in cases["cases"]:
        validate_request(c["request"])  # every demo case is wire-valid


def test_server_image_request_end_to_end(server):
    body = {"state": img_state(), "model": "kapteeni-v1.1c",
            "questions": {"q": {"type": "noul", "instructions": "red?",
                                "criteria": {"true": "red", "false": "no"}}}}
    status, resp = post(server, body)
    assert status == 200
    assert resp["model"] == "kapteeni-v1.1c"
    a = resp["answers"]["q"]
    assert a["type"] == "noul" and 0.0 <= a["noul"] <= 1.0
    assert resp["usage"]["input_tokens"] > 0


def test_from_dist_roundtrip(tmp_path):
    """pack_v11c's layout round-trips torch-free: config + heads
    safetensors -> from_dist -> constructor paths + temperatures (the
    torch model itself loads lazily at serve time)."""
    from safetensors.torch import save_file
    from kapteeni.heads import PassMLP
    d = tmp_path / "dist"
    d.mkdir()
    (d / "kapteeni-config.json").write_text(json.dumps({
        "served_as": "kapteeni-v1.1c", "in_dim": 4,
        "head_temperatures": {"noul": 1.2, "choice": 0.9, "score": 1.8}}))
    head = PassMLP(4)
    tensors = {}
    for qt in ("noul", "choice", "score"):
        for k, v in head.state_dict().items():  # clone: one storage
            tensors[f"{qt}.{k}"] = v.clone()     # shared 3x otherwise
    save_file(tensors, str(d / "heads.safetensors"))
    from kapteeni.model_v11c import SystemOneV11C
    m = SystemOneV11C.from_dist(str(d))
    assert m.temps == {"noul": 1.2, "choice": 0.9, "score": 1.8}
    assert m.in_dim == 4 and m.adapter == ""  # merged-model path
    assert not m._loaded  # nothing heavy happened during construction


def test_server_bad_image_422(server):
    body = {"state": {"image": "!!!not-base64!!!", "note": "x"},
            "model": "jev-latest",
            "questions": {"q": {"type": "noul", "instructions": "x",
                                "criteria": None}}}
    status, resp = post(server, body)
    assert status == 422 and resp["error"]["field"] == "state.image"


def test_server_unknown_model_422(server):
    status, resp = post(server, {"state": "s", "model": "nope",
                                 "questions": {"q": {
                                     "type": "noul", "instructions": "x"}}})
    assert status == 422 and resp["error"]["field"] == "model"


def test_server_demo_path_traversal_forbidden(server):
    try:
        with urllib.request.urlopen(server + "/demo/../contract.py") as r:
            status = r.status
    except urllib.error.HTTPError as e:
        status = e.code
    assert status in (403, 404)