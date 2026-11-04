"""
list_cameras.py — 이 컴퓨터에서 열리는 카메라를 찾고, 쓸 카메라를 고른다.

실행 (프로젝트 루트에서):
    python setup/list_cameras.py           열리는 카메라 번호·이름 보기 (번호별 미리보기 창도 띄움)
    python setup/list_cameras.py --set 1   1번 카메라를 쓰도록 저장 (camera.local)

고른 번호는 config.CAM_INDEX가 읽으므로 server.py, realtime.py, collect.py,
check_env.py 모두 같은 카메라를 쓴다. 서버가 켜져 있으면 끄고 다시 켜야 적용된다.
"""

import argparse
import ctypes
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import cv2                                          # noqa: E402

from config import CAM_FILE, CAM_INDEX               # noqa: E402

# 없는 번호를 열어 볼 때 OpenCV가 찍는 경고·오류 메시지를 숨긴다.
# FFMPEG 경고 한 줄은 이것으로 안 꺼져서 run_test.bat가 OPENCV_LOG_LEVEL=OFF로 실행한다.
cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)

MAX_INDEX = 6     # 0~5번까지 찾아본다
WARMUP_READS = 5  # 첫 몇 장은 노출이 덜 맞아 어둡게 나와서 버린다
THUMB = (320, 240)
PREVIEW_WIN = "SOAR camera select"


def probe(index):
    """열리고 프레임이 읽히면 마지막 프레임, 아니면 None.
    landmarker.open_camera와 같은 순서(DSHOW -> 기본)로 열어 봐야 여기서 고른 번호가 실제로도 열린다."""
    backends = []
    if sys.platform.startswith("win"):
        backends.append(cv2.CAP_DSHOW)
    backends.append(cv2.CAP_ANY)

    for be in backends:
        cap = cv2.VideoCapture(index, be)
        try:
            if not cap.isOpened():
                continue
            frame = None
            for _ in range(WARMUP_READS):
                ok, f = cap.read()
                if ok and f is not None:
                    frame = f
            if frame is not None:
                return frame
        finally:
            cap.release()
    return None


# DirectShow COM 인터페이스 ID (strmif.h, ocidl.h)
CLSID_SYSTEM_DEVICE_ENUM = "62BE5D10-60EB-11d0-BD3B-00A0C911CE86"
IID_ICREATE_DEV_ENUM = "29840822-5B84-11D0-BD3B-00A0C911CE86"
CLSID_VIDEO_INPUT_CATEGORY = "860BB310-5D01-11d0-BD3B-00A0C911CE86"
IID_IPROPERTY_BAG = "55272A00-42CB-11CE-8135-00AA004BB851"


def _guid(text):
    buf = (ctypes.c_ubyte * 16)()
    ctypes.memmove(buf, uuid.UUID(text).bytes_le, 16)
    return buf


def _com(obj, slot, *argtypes):
    """COM 객체 obj의 vtable slot번째 메서드를 부를 수 있게 만든다."""
    vtbl = ctypes.cast(obj, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
    fn = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, *argtypes)(vtbl[slot])
    return lambda *args: fn(obj, *args)


def _release(obj):
    if obj:
        _com(obj, 2)()   # IUnknown::Release


class _Variant(ctypes.Structure):
    _fields_ = [("vt", ctypes.c_ushort), ("reserved", ctypes.c_ushort * 3),
                ("val", ctypes.c_void_p), ("pad", ctypes.c_void_p)]


def device_names():
    """DirectShow 카메라 이름을 번호 순서대로. OpenCV의 CAP_DSHOW 번호가 이 목록 순서를 따른다.
    추가 패키지 없이 ctypes로 윈도우 COM을 직접 부른다. 실패하면 빈 목록."""
    if not sys.platform.startswith("win"):
        return []
    try:
        ole32 = ctypes.windll.ole32
        oleaut32 = ctypes.windll.oleaut32
    except (AttributeError, OSError):
        return []

    names = []
    inited = ole32.CoInitialize(None) >= 0
    dev_enum = ctypes.c_void_p()
    enum = ctypes.c_void_p()
    try:
        if ole32.CoCreateInstance(_guid(CLSID_SYSTEM_DEVICE_ENUM), None, 1,   # CLSCTX_INPROC_SERVER
                                  _guid(IID_ICREATE_DEV_ENUM), ctypes.byref(dev_enum)) != 0:
            return []
        # ICreateDevEnum::CreateClassEnumerator. 카메라가 하나도 없으면 S_FALSE(1)
        create = _com(dev_enum, 3, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong)
        if create(_guid(CLSID_VIDEO_INPUT_CATEGORY), ctypes.byref(enum), 0) != 0:
            return []
        next_ = _com(enum, 3, ctypes.c_ulong, ctypes.c_void_p, ctypes.c_void_p)   # IEnumMoniker::Next
        while True:
            mon = ctypes.c_void_p()
            fetched = ctypes.c_ulong()
            if next_(1, ctypes.byref(mon), ctypes.byref(fetched)) != 0 or not fetched.value:
                break
            name = "(이름 없음)"
            bag = ctypes.c_void_p()
            # IMoniker::BindToStorage -> IPropertyBag::Read("FriendlyName")
            bind = _com(mon, 9, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)
            if bind(None, None, _guid(IID_IPROPERTY_BAG), ctypes.byref(bag)) == 0:
                var = _Variant()
                read = _com(bag, 3, ctypes.c_wchar_p, ctypes.c_void_p, ctypes.c_void_p)
                if read("FriendlyName", ctypes.byref(var), None) == 0 and var.vt == 8:   # VT_BSTR
                    name = ctypes.wstring_at(var.val)
                oleaut32.VariantClear(ctypes.byref(var))
                _release(bag)
            _release(mon)
            names.append(name)
    except OSError:
        return []
    finally:
        _release(enum)
        _release(dev_enum)
        if inited:
            ole32.CoUninitialize()
    return names


def show_preview(frames, names):
    """번호별 한 장씩 이름과 함께 나란히 띄운다. 이름이 같은 카메라 두 대도 화면으로 구분할 수 있게."""
    tiles = []
    for i, frame in frames.items():
        tile = cv2.resize(frame, THUMB)
        label = f"[{i}]" + (" now" if i == CAM_INDEX else "")
        cv2.rectangle(tile, (0, 0), (THUMB[0], 64), (0, 0, 0), -1)
        cv2.putText(tile, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        if i < len(names):
            # putText는 한글을 못 그리지만 장치 이름은 보통 영문이다
            cv2.putText(tile, names[i][:32], (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        tiles.append(tile)
    try:
        cv2.imshow(PREVIEW_WIN, cv2.hconcat(tiles))
        print("미리보기 창에서 번호를 확인하고, 창을 닫거나 아무 키나 누르세요.")
        # 창의 X 버튼으로 닫아도 빠져나오도록 waitKey(0) 대신 짧게 반복한다
        while cv2.waitKey(100) == -1:
            if cv2.getWindowProperty(PREVIEW_WIN, cv2.WND_PROP_VISIBLE) < 1:
                break
        cv2.destroyAllWindows()
    except cv2.error:
        pass   # 창을 띄울 수 없는 환경이면 번호 목록만 쓴다


def list_cameras(preview=True):
    print("카메라를 찾는 중... (번호마다 1~2초 걸릴 수 있음)")
    names = device_names()
    frames = {}
    for i in range(MAX_INDEX):
        frame = probe(i)
        if frame is None:
            continue
        frames[i] = frame
        mark = "  <- 지금 사용 중" if i == CAM_INDEX else ""
        name = names[i] if i < len(names) else "(이름 모름)"
        print(f"  [{i}] {name}  {frame.shape[1]}x{frame.shape[0]}{mark}")
    found = list(frames)

    if not found:
        print("열리는 카메라가 없습니다. 줌·팀즈 등 카메라를 쓰는 프로그램과 SOAR 서버를 끄고 다시 해 보세요.")
        return found

    if CAM_INDEX not in found:
        print(f"[주의] 지금 설정된 {CAM_INDEX}번 카메라는 열리지 않습니다.")
    if preview:
        show_preview(frames, names)
    return found


def set_camera(index):
    if probe(index) is None:
        print(f"[FAIL] {index}번 카메라를 열 수 없어 저장하지 않았습니다.")
        return 1
    CAM_FILE.write_text(f"{index}\n", encoding="utf-8")
    print(f"[ OK ] 이제 {index}번 카메라를 씁니다. (저장 위치: {CAM_FILE.name})")
    print("       서버가 켜져 있다면 끄고 다시 켜야 적용됩니다.")
    return 0


def main():
    ap = argparse.ArgumentParser(description="카메라 목록 보기 / 쓸 카메라 고르기")
    ap.add_argument("--set", type=int, metavar="N", help="N번 카메라를 쓰도록 저장")
    ap.add_argument("--no-preview", action="store_true", help="미리보기 창 없이 번호만 보기")
    args = ap.parse_args()
    if args.set is not None:
        return set_camera(args.set)
    return 0 if list_cameras(preview=not args.no_preview) else 1


if __name__ == "__main__":
    raise SystemExit(main())
