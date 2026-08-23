"""Rubrik deterministik: label emas per-turn -> target penilaian sesi (mode Pelatih).

KENAPA RUBRIK, BUKAN LLM LAGI?
Dialog hasil generate sudah punya label emas per turn (`teknik`, `kualitas`). Target
sesi (`skor_total`, `per_teknik`, `saran`) diturunkan dari label itu lewat rubrik
tetap di bawah — BUKAN dengan memanggil LLM sekali lagi. Konsekuensinya:

  * Target konsisten & bisa diaudit. Rubrik yang sama dipakai `4_evaluate.py` untuk
    membuat skor acuan, jadi metrik evaluasi tidak bergantung pada mood generator.
  * Yang dipelajari model = pemetaan "percakapan -> diagnosis", bukan meniru gaya
    tulis LLM guru. Ini memperkecil risiko "model hanya sepandai LLM-nya" (PLAN.md §6).

BATASAN YANG DIAKUI:
Kalimat `catatan` & `saran` berasal dari bank teks rubrik, jadi variasi bahasanya
terbatas. Model belajar MEMILIH diagnosis yang tepat, bukan mengarang saran baru.
Untuk MVP ini disengaja: penilaian yang stabil lebih berguna bagi trainee daripada
saran yang indah tapi berubah-ubah.

Dipakai oleh: 2_prepare_sft.py (buat target latih) & 4_evaluate.py (skor acuan).
"""
from __future__ import annotations

# Bobot kontribusi tiap teknik ke skor total. Teknik inti konsultatif (gali kebutuhan,
# presentasi manfaat, atasi keberatan) berbobot paling besar; upsell kecil karena tidak
# selalu wajar muncul di setiap percakapan.
BOBOT = {
    "sapa_rapport": 1.0,
    "gali_kebutuhan": 2.0,
    "presentasi_manfaat": 2.0,
    "atasi_keberatan": 2.0,
    "closing": 1.5,
    "upsell": 0.5,
}

# Skor saat sebuah teknik sama sekali tidak muncul. Bukan 0: tidak memakai upsell
# bukan kesalahan fatal, tapi tetap peluang yang terlewat.
SKOR_ABSEN = 50

_CATATAN = {
    "absen": "Teknik ini belum terlihat sama sekali di percakapan.",
    "semua_baik": "Diterapkan konsisten dan tepat sasaran.",
    "campur": "Sudah muncul, tapi sebagian giliran masih lemah.",
    "semua_lemah": "Sudah dicoba, tapi eksekusinya masih lemah.",
}

# Bank saran: (teknik, kasus) -> kalimat coaching konkret.
_SARAN = {
    "sapa_rapport": {
        "absen": "Buka percakapan dengan sapaan hangat dan perkenalan singkat sebelum masuk ke produk — pembeli perlu merasa dilayani, bukan diburu.",
        "lemah": "Sapaan masih terasa buru-buru. Sebut sapaan hormat (Pak/Bu), tanya kabar atau keperluannya dulu, baru arahkan ke produk.",
    },
    "gali_kebutuhan": {
        "absen": "Belum ada pertanyaan yang menggali kebutuhan. Sebelum menawarkan, tanyakan situasi pemakaian (untuk siapa, seberapa sering, budget kisaran) — ini yang membuat rekomendasi terasa pas.",
        "lemah": "Pertanyaannya masih dangkal atau tertutup. Pakai urutan SPIN: situasi → masalah → dampak masalah itu → kebutuhan yang muncul.",
    },
    "presentasi_manfaat": {
        "absen": "Produk belum dijelaskan manfaatnya. Sampaikan minimal satu keunggulan yang langsung menjawab kebutuhan yang tadi disebut pelanggan.",
        "lemah": "Penjelasan masih berupa daftar fitur. Ubah jadi manfaat: 'fitur X → artinya Bapak/Ibu dapat Y' dan kaitkan ke kebutuhan yang tadi disebutkan.",
    },
    "atasi_keberatan": {
        "absen": "Keberatan pelanggan belum ditanggapi secara khusus. Akui dulu keberatannya, baru jawab dengan nilai atau bukti.",
        "lemah": "Saat ditolak, hindari langsung memberi diskon atau membela diri. Pola yang lebih kuat: akui → klarifikasi maksudnya → jawab dengan bukti/nilai → cek apakah sudah lega.",
    },
    "closing": {
        "absen": "Percakapan berakhir tanpa ajakan konkret. Tutup dengan langkah berikutnya yang jelas — simulasi cicilan, test drive, atau jadwal pengiriman.",
        "lemah": "Ajakan menutupnya masih menekan atau menggantung. Tawarkan pilihan ('mau yang A atau B?') supaya pelanggan tetap merasa memutuskan sendiri.",
    },
    "upsell": {
        "absen": "Belum ada tawaran pelengkap. Bila relevan, tawarkan satu tambahan yang benar-benar menambah nilai (paket servis, perlindungan, aksesori) — cukup satu, jangan memborong.",
        "lemah": "Tawaran tambahannya belum terasa relevan. Kaitkan ke kebutuhan yang sudah tergali, dan sebutkan nilainya, bukan sekadar menambah biaya.",
    },
}


def _skor_teknik(n_total: int, n_baik: int) -> int:
    """Skor 0-100 satu teknik dari jumlah giliran & berapa yang berkualitas 'baik'."""
    if n_total == 0:
        return SKOR_ABSEN
    rasio = n_baik / n_total
    skor = round(55 + 40 * rasio)  # 55 (semua lemah) .. 95 (semua baik)
    if rasio == 1.0 and n_total >= 2:
        skor = min(100, skor + 5)  # bonus konsistensi: baik dan diulang
    return skor


def _kasus(n_total: int, n_baik: int) -> str:
    if n_total == 0:
        return "absen"
    if n_baik == n_total:
        return "semua_baik"
    if n_baik == 0:
        return "semua_lemah"
    return "campur"


def nilai_sesi(turns: list[dict], teknik_ids: list[str], maks_saran: int = 3) -> dict:
    """Turunkan target mode Pelatih dari label emas seluruh dialog.

    Return dict yang bentuknya PERSIS `EvaluateResponse` di backend/app/schemas.py:
    {"skor_total": int, "per_teknik": [{"teknik", "skor", "catatan"}], "saran": [str]}
    """
    hitung = {t: {"total": 0, "baik": 0} for t in teknik_ids}
    for t in turns:
        if t.get("speaker") != "sales":
            continue
        tek = t.get("teknik")
        if tek not in hitung:
            continue
        hitung[tek]["total"] += 1
        if t.get("kualitas") == "baik":
            hitung[tek]["baik"] += 1

    per_teknik = []
    kandidat_saran: list[tuple[float, str]] = []
    for tek in teknik_ids:
        n, nb = hitung[tek]["total"], hitung[tek]["baik"]
        skor = _skor_teknik(n, nb)
        kasus = _kasus(n, nb)
        per_teknik.append({"teknik": tek, "skor": skor, "catatan": _CATATAN[kasus]})

        if kasus == "semua_baik":
            continue
        # Prioritas saran = seberapa besar kekurangannya, ditimbang pentingnya teknik.
        prioritas = BOBOT[tek] * (100 - skor)
        kunci = "absen" if kasus == "absen" else "lemah"
        kandidat_saran.append((prioritas, _SARAN[tek][kunci]))

    total_bobot = sum(BOBOT[p["teknik"]] for p in per_teknik)
    skor_total = round(sum(BOBOT[p["teknik"]] * p["skor"] for p in per_teknik) / total_bobot)

    kandidat_saran.sort(key=lambda x: -x[0])
    saran = [s for _, s in kandidat_saran[:maks_saran]]
    if not saran:
        saran = ["Seluruh teknik inti sudah dijalankan dengan baik — pertahankan dan latih kecepatan membaca sinyal beli."]

    return {"skor_total": skor_total, "per_teknik": per_teknik, "saran": saran}
