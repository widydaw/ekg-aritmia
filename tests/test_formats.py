"""Uji pembacaan format berkas EKG unggahan."""
import io
import struct
from pathlib import Path

import numpy as np
import pytest

from src.file_formats import default_channel, read_upload, to_record
from src.synthetic import sample_record


@pytest.fixture(scope="module")
def ecg():
    rec = sample_record("SIN-02", duration=20)   # 360 Hz, mV, dengan anotasi
    return rec


def _resample(x, fs_in, fs_out):
    t = np.arange(int(len(x) / fs_in * fs_out)) / fs_out
    return np.interp(t, np.arange(len(x)) / fs_in, x)


def test_wfdb_with_annotations(tmp_path, ecg):
    import wfdb
    sig = np.column_stack([ecg.signal, -ecg.signal])
    wfdb.wrsamp("rek1", fs=360, units=["mV", "mV"], sig_name=["MLII", "V5"], p_signal=sig, fmt=["16", "16"],
                write_dir=str(tmp_path))
    wfdb.wrann("rek1", "atr", ecg.r_peaks, symbol=["N"] * len(ecg.r_peaks), write_dir=str(tmp_path))
    files = [(p.name, p.read_bytes()) for p in tmp_path.iterdir()]
    up = read_upload(files)
    assert up.fmt == "WFDB" and set(up.channels) == {"MLII", "V5"}
    assert default_channel(up) == "MLII"
    rec = to_record(up, "MLII")
    assert rec.has_annotations and len(rec.r_peaks) == len(ecg.r_peaks)
    assert np.allclose(rec.signal[:1000], ecg.signal[:1000], atol=0.01)


def test_wfdb_missing_dat(tmp_path, ecg):
    import wfdb
    wfdb.wrsamp("rek2", fs=360, units=["mV"], sig_name=["MLII"], p_signal=ecg.signal[:, None], fmt=["16"],
                write_dir=str(tmp_path))
    with pytest.raises(ValueError, match="belum diunggah"):
        read_upload([("rek2.hea", (tmp_path / "rek2.hea").read_bytes())])


def _edf_bytes(x_uv, fs, label="ECG II"):
    n_rec, dur = int(len(x_uv) // fs), 1
    ns, nsamp = 1, int(fs)
    hdr = (f"{'0':<8}{'pasien':<80}{'rekaman':<80}{'01.01.26':<8}{'00.00.00':<8}{256 * (1 + ns):<8}{'':<44}"
           f"{n_rec:<8}{dur:<8}{ns:<4}").encode("ascii")
    hdr += f"{label:<16}".encode() + f"{'AgAgCl':<80}".encode() + f"{'uV':<8}".encode()
    hdr += f"{-5000:<8}".encode() + f"{5000:<8}".encode() + f"{-32768:<8}".encode() + f"{32767:<8}".encode()
    hdr += f"{'':<80}".encode() + f"{nsamp:<8}".encode() + f"{'':<32}".encode()
    dig = np.round((x_uv[: n_rec * nsamp] + 5000) / 10000 * 65535 - 32768).astype("<i2")
    return hdr + dig.tobytes()


def test_edf(ecg):
    x = _resample(ecg.signal, 360, 250) * 1000  # µV, 250 Hz
    up = read_upload([("rek.edf", _edf_bytes(x, 250))])
    assert up.fmt == "EDF" and list(up.channels) == ["ECG II"]
    sig, fs = up.channels["ECG II"]
    assert fs == 250 and abs(np.max(sig) - np.max(x / 1000)) < 0.01  # dikonversi ke mV
    rec = to_record(up, "ECG II")
    assert rec.fs == 360 and abs(rec.duration - 20) < 0.1


def test_hl7_aecg(ecg):
    x = _resample(ecg.signal, 360, 500)
    digits = " ".join(str(int(v)) for v in np.round(x * 1000 / 2.5))
    xml = f"""<AnnotatedECG xmlns="urn:hl7-org:v3" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
<component><series><component><sequenceSet>
 <component><sequence><code code="TIME_ABSOLUTE"/><value xsi:type="GLIST_TS"><head value="20260101"/><increment value="0.002" unit="s"/></value></sequence></component>
 <component><sequence><code code="MDC_ECG_LEAD_II"/><value xsi:type="SLIST_PQ"><origin value="0" unit="uV"/><scale value="2.5" unit="uV"/><digits>{digits}</digits></value></sequence></component>
</sequenceSet></component></series></component></AnnotatedECG>"""
    up = read_upload([("ekg.xml", xml.encode())])
    assert list(up.channels) == ["II"]
    sig, fs = up.channels["II"]
    assert fs == pytest.approx(500) and np.allclose(sig, x, atol=0.003)


def test_mat(ecg):
    from scipy.io import savemat
    buf = io.BytesIO()
    savemat(buf, {"val": (np.vstack([ecg.signal, ecg.signal]) * 1000).astype(np.int16)})
    up = read_upload([("A0001.mat", buf.getvalue())], fs_hint=360)
    assert set(up.channels) == {"Kanal 1", "Kanal 2"}
    assert np.allclose(up.channels["Kanal 1"][0], ecg.signal, atol=0.002)   # µV dianggap, dibagi 1000


def test_csv_multi_with_time(ecg):
    x = _resample(ecg.signal, 360, 500)
    t = np.arange(len(x)) / 500
    text = "time,I,II\n" + "\n".join(f"{a:.4f},{b * 0.5:.4f},{b:.4f}" for a, b in zip(t, x))
    up = read_upload([("chapman.csv", text.encode())], fs_hint=360)
    assert list(up.channels) == ["I", "II"]
    assert up.channels["II"][1] == pytest.approx(500)   # fs dari kolom waktu
    assert default_channel(up) == "II"


def test_csv_errors_and_long():
    with pytest.raises(ValueError, match="kolom"):
        read_upload([("a.csv", b"1;2\n3\n" * 3000)])
    with pytest.raises(ValueError, match="Ekstensi"):
        read_upload([("a.json", b"{}")])
    with pytest.raises(ValueError, match=r"\.hea"):
        read_upload([("100.dat", b"\x00" * 10)])
    long = "\n".join(["0.1", "0.2"] * (360 * 1000))  # 2000 s > 30 menit
    rec = to_record(read_upload([("p.csv", long.encode())], 360), "Sinyal")
    assert abs(rec.duration - 1800) < 1 and "dipotong" in rec.description
