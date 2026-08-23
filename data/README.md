# Dataset — Dialog Jualan Sintetik (JagoJual)

Dataset dialog jualan toko offline berbahasa Indonesia untuk melatih **mode Pelatih**
(penilaian teknik jualan) dan menyeed **mode Pelanggan** (role-play). Dibuat tim; sintetik, di-*grounding* ke
framework sales nyata, dan divalidasi otomatis terhadap schema serta aturan bisnis.
Lihat [`PLAN.md`](../PLAN.md) untuk rasional lengkap.

## Kenapa sintetik?

Tidak ada dataset publik dialog jualan retail offline ber-Bahasa Indonesia. Korpus terdekat
berbahasa Inggris & domain call-center (mis. **TeleSalesCorpus**, yang taksonomi turn-nya —
GREET → DISCOVER_NEEDS → PITCH_BENEFIT → HANDLE_OBJECTION → CLOSE_CALL — kami adopsi & sesuaikan).
Rulebook mengizinkan data sintetik; untuk domain seniche ini, sintetik yang dibuat sesuai konteks
lebih relevan daripada memaksakan data domain lain.

## Taksonomi (sumber kebenaran: [`taxonomy.json`](taxonomy.json))

Selaras dengan `backend/app/schemas.py` — **jangan** hardcode label di tempat lain, baca file ini.

- **Teknik sales (6, label turn `sales`):** `sapa_rapport` · `gali_kebutuhan` · `presentasi_manfaat` ·
  `atasi_keberatan` · `closing` · `upsell`
- **Kualitas turn `sales` (2):** `baik` · `lemah` (untuk kontras "sales kuat vs lemah")
- **Keberatan global (6, label turn `pelanggan`):** `harga` · `bandingkan_kompetitor` · `ragu_kualitas` ·
  `tidak_butuh` · `mau_pikir_pikir` · `cuma_lihat_lihat`
- **Topik keberatan per bidang** (mis. `boros_bbm`, `dp_cicilan`, `garansi`, `harga_toko_sebelah`) =
  manifestasi konkret yang **dipetakan** ke salah satu keberatan global. Dipakai menyeed variasi
  skenario, bukan label klasifikasi terpisah. (Jadi label `keberatan` di dialog selalu salah satu
  dari 6 global — topik hanya hidup di `taxonomy.json` & `scenario_matrix.json`.)
- **Persona (6):** `skeptis` · `buru_buru` · `sensitif_harga` · `banyak_tanya` · `loyal_merk_lain` ·
  `cuma_lihat`. `emosi_awal` hanya metadata (kepala emosi dibuang dari MVP).

Grounding label: **SPIN**, **AIDA**, **FAB**, objection-handling.

## Struktur

```
data/
├── taxonomy.json          # sumber kebenaran tunggal semua label
├── scenario_matrix.json   # daftar sel yang digenerate (hasil build_scenario_matrix.py)
├── dialogs/
│   ├── schema.json        # JSON Schema level dialog (Draft-07)
│   ├── contoh_*.json      # 2 seed manual tervalidasi (juga smoke-test)
│   └── <scenario_id>.json # dialog hasil generate
├── sft/                   # TURUNAN, tidak di-commit (hasil 2_prepare_sft.py)
└── README.md
```

Dialog level sumber ini di-*flatten* jadi contoh SFT oleh
[`../model/training/2_prepare_sft.py`](../model/training/2_prepare_sft.py) menjadi format chat:

```json
{"messages": [
  {"role": "system", "content": "Kamu PELATIH sales. ..."},
  {"role": "user", "content": "Konteks: ...\n\nPercakapan:\nSales: ...\nPelanggan: ...\n\nNilai percakapan di atas."},
  {"role": "assistant", "content": "{\"skor_total\": 91, \"per_teknik\": [...], \"saran\": [...]}"}
]}
```

Target penilaian sesi diturunkan dari label emas per-turn lewat **rubrik deterministik**
(`model/training/rubric.py`), bukan dengan memanggil LLM kedua kali — lihat README di
folder itu untuk rasional dan batasannya.

**Aturan anotasi:** satu giliran `sales` = satu teknik. Ucapan yang sekaligus closing dan
upsell dipecah jadi dua giliran berurutan; kalau digabung, penilaian sesi akan menyimpulkan
salah satu teknik "tidak dipakai" padahal terlihat di transkrip.

## Cara membuat (reproduksi)

Skrip di [`../model/training/`](../model/training/). Ringkas:

```bash
# 1) rencanakan sel seimbang (~150/bidang = 300 dialog)
python build_scenario_matrix.py --per-bidang 150

# 2a) generator lokal deterministik — INI yang menghasilkan 300 dialog di data/dialogs/
#     (grounding.generator = "grounded-local-v2"). Tanpa API, siapa pun bisa mengulang
#     dan hasilnya sama.
python 1_generate_local.py

# 2b) jalur alternatif lewat LLM eksternal (OpenAI-compatible). Ada di repo tapi TIDAK
#     dipakai untuk dataset yang dikirimkan.
export JAGO_LLM_BASE_URL=...   JAGO_LLM_API_KEY=...   JAGO_LLM_MODEL=...
python 1_generate_data.py --dry-run     # cek prompt
python 1_generate_data.py --limit 5     # smoke test
python 1_generate_data.py               # semua (resume: skip yang sudah ada)

# 3) validasi semua file (schema + aturan bisnis)
python 1_generate_data.py --validate-only

# 4) flatten jadi contoh SFT (train/val/test + stats)
python 2_prepare_sft.py
```

## Jaga mutu

**Yang sudah dijalankan:**

1. **Matriks terkontrol** — distribusi bidang, topik, persona, dan keberatan diseimbangkan sejak
   awal lewat `build_scenario_matrix.py`, bukan diserahkan ke kebetulan saat generate.
2. **Grounding eksplisit** — tiap dialog dibangun mengikuti kerangka SPIN, AIDA, FAB, dan pola
   objection-handling; jejaknya tersimpan di `grounding.framework` tiap berkas.
3. **Validasi otomatis** — schema, label per pembicara, dan kehadiran keberatan utama dicek untuk
   seluruh 300 dialog sebelum masuk dataset.

**Yang belum dijalankan, dan akibatnya:**

4. **Validasi praktisi** — rencananya sebagian dialog di-spot-check oleh orang berpengalaman jualan
   lalu labelnya dikoreksi. Ini **tidak sempat kami kerjakan**. Silakan cek berkas mana pun:
   `grounding.divalidasi_manusia` bernilai `false` untuk 298 dari 300 dialog; yang `true` hanya dua
   contoh tulisan tangan.
5. **Seed transkrip asli** — belum ada. Seluruh dialog murni bangkitan.

Akibatnya keragaman bahasa dataset ini masih terikat pola template. Itu sebabnya angka evaluasi
model kami tinggi, dan kenapa angka itu sebaiknya dibaca sebagai lompatan relatif terhadap base
model, bukan sebagai perkiraan performa di percakapan dunia nyata.

## Kejujuran, lisensi & etika (untuk proposal)

- Label berasal dari satu generator yang kami tulis sendiri, jadi ada risiko model hanya sepandai
  aturan generator itu. Yang menahannya sekarang: grounding ke framework sales dan validasi
  otomatis. Yang seharusnya menahannya lebih jauh, yaitu validasi praktisi dan transkrip asli,
  belum ada.
- Data sintetik dibuat tim → bebas dipakai untuk lomba. Transkrip asli **wajib dianonimkan**
  (tanpa nama, nomor, identitas) sebelum masuk repo.
- Framework grounding (SPIN/AIDA/FAB) dipakai sebagai *acuan teknik*, bukan penyalinan teks.
