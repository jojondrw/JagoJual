"""Uji bagian MODE=local yang tidak butuh GPU: prompt & normalisasi keluaran model.

Pemuatan model sengaja tidak diuji di sini — itu butuh GPU dan bobot base 3B. Yang diuji
adalah lapisan yang paling mungkin salah diam-diam: format prompt (kontrak dengan data
latih) dan pembersihan keluaran model sebelum masuk ke respons API.
"""

import pytest

from app import prompts
from app.llm import LLMError, _parse_json, normalisasi_penilaian
from app.scenarios import get_scenario
from app.schemas import ChatMessage

SKENARIO = get_scenario("otomotif_boros")
RIWAYAT = [
    ChatMessage(role="pelanggan", text="Mobil segini boros nggak sih?"),
    ChatMessage(role="sales", text="Boleh tahu sehari berapa kilometer, Pak?"),
]


# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #

def test_transkrip_pakai_label_pembicara():
    assert prompts.render_transcript(RIWAYAT) == (
        "Pelanggan: Mobil segini boros nggak sih?\n"
        "Sales: Boleh tahu sehari berapa kilometer, Pak?"
    )


def test_prompt_pelatih_memuat_konteks_dan_transkrip():
    isi = prompts.coach_session_user(SKENARIO, RIWAYAT)
    assert "SUV 1.5L Turbo" in isi
    assert "Showroom / dealer mobil." in isi
    assert "Sales: Boleh tahu sehari berapa kilometer, Pak?" in isi
    assert isi.rstrip().endswith("Nilai percakapan di atas.")


def test_system_pelatih_pakai_kurung_tunggal():
    # Kalau ganda, model dilatih/diminta menulis "{{" dan keluarannya gagal di-parse.
    assert '{"skor_total": int' in prompts.COACH_SYSTEM
    assert "{{" not in prompts.COACH_SYSTEM


def test_prompt_roleplay_memuat_persona():
    isi = prompts.roleplay_system(SKENARIO)
    assert "skeptis-hemat" in isi and "ragu" in isi and "SUV 1.5L Turbo" in isi


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "mentah",
    [
        '{"skor_total": 80}',
        '```json\n{"skor_total": 80}\n```',
        'Tentu, berikut penilaiannya: {"skor_total": 80} Semoga membantu.',
    ],
)
def test_parse_json_toleran_pembungkus(mentah):
    assert _parse_json(mentah) == {"skor_total": 80}


@pytest.mark.parametrize("mentah", ["maaf saya tidak bisa", "[1, 2, 3]", ""])
def test_parse_json_menolak_yang_bukan_objek(mentah):
    assert _parse_json(mentah) is None


# --------------------------------------------------------------------------- #
# Normalisasi
# --------------------------------------------------------------------------- #

def test_normalisasi_lengkap():
    hasil = normalisasi_penilaian(
        {
            "skor_total": 82,
            "per_teknik": [
                {"teknik": "closing", "skor": 70, "catatan": "ok"},
                {"teknik": "sapa_rapport", "skor": 90, "catatan": "ramah"},
            ],
            "saran": ["a", "b", "c", "d"],
        }
    )
    assert hasil.skor_total == 82
    # Diurutkan sesuai taksonomi, bukan urutan keluaran model.
    assert [t.teknik for t in hasil.per_teknik] == ["sapa_rapport", "closing"]
    assert hasil.saran == ["a", "b", "c"]  # dibatasi 3


def test_normalisasi_buang_teknik_asing_dan_duplikat():
    hasil = normalisasi_penilaian(
        {
            "skor_total": 50,
            "per_teknik": [
                {"teknik": "closing", "skor": 60},
                {"teknik": "closing", "skor": 99},  # duplikat -> yang pertama menang
                {"teknik": "menghipnotis_pembeli", "skor": 100},  # di luar taksonomi
                {"teknik": "upsell", "skor": "bukan angka"},
            ],
            "saran": [],
        }
    )
    assert [(t.teknik, t.skor) for t in hasil.per_teknik] == [("closing", 60)]


def test_normalisasi_batasi_skor_ke_rentang():
    hasil = normalisasi_penilaian(
        {"skor_total": 999, "per_teknik": [{"teknik": "closing", "skor": -20}], "saran": []}
    )
    assert hasil.skor_total == 100
    assert hasil.per_teknik[0].skor == 0


def test_skor_total_dihitung_bila_model_lupa():
    hasil = normalisasi_penilaian(
        {
            "per_teknik": [
                {"teknik": "closing", "skor": 60},
                {"teknik": "upsell", "skor": 80},
            ],
            "saran": [],
        }
    )
    assert hasil.skor_total == 70


@pytest.mark.parametrize(
    "obj",
    [
        None,
        {},
        {"skor_total": 80, "per_teknik": [], "saran": []},
        {"skor_total": 80, "per_teknik": [{"teknik": "tidak_ada", "skor": 90}], "saran": []},
    ],
)
def test_keluaran_tak_terpakai_jadi_error_bukan_skor_karangan(obj):
    # Lebih baik 503 yang jujur daripada angka yang dikarang backend.
    with pytest.raises(LLMError):
        normalisasi_penilaian(obj)
