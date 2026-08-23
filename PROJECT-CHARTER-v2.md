# Piagam Proyek v2 — Multicamera Capacity Study

**Arka · 22 Agustus 2026 · menggantikan v1**
Dokumen keputusan, bahasa Indonesia. **Isi repo seluruhnya bahasa Inggris.**

Nama repo — putuskan sekarang, jangan setelah commit pertama:
`multicam-bench` · `streamscope` · `camfit` · `ingest-bench`

---

## 1. Definisi

> Sebuah pipeline video multicamera yang bisa dijalankan terus-menerus, **beserta alat
> ukur yang menentukan berapa banyak kamera yang muat di sebuah mesin.**

Dua kalimat, satu proyek. Pipeline adalah **objek yang diukur**; harness adalah
**alat ukurnya**. Kalau salah satu dibuang, yang lain kehilangan makna.

---

## 2. Prinsip yang mengikat seluruh desain

> **Apa pun yang tidak kita ketahui menjadi parameter, bukan asumsi.**

Kita tidak tahu resolusi target, fps deteksi, codec, jumlah kamera, atau mesin
tujuan. Kita **tidak menebaknya dan tidak menanyakannya** — kita menjadikannya sumbu
pengukuran dan melaporkan seluruh permukaannya.

| Tidak diketahui | Sumbu |
|---|---|
| fps deteksi | 1, 2, 5, 10, 25 |
| resolusi | 360p, 720p, 1080p |
| codec | H.264, H.265 |
| jumlah stream | 1 → sampai patah |
| backend decode | cpu, qsv, d3d11va, nvdec |
| mesin | koefisien terkalibrasi per mesin |
| tujuan output | sink yang bisa diganti |

Konsekuensi: hasilnya bukan satu angka, melainkan **permukaan kapasitas**. Titik
mana pun yang dibutuhkan pembaca sudah ada di dalamnya.

---

## 3. Tiga lapisan

### Lapisan 1 — JALAN (pipeline)

```
RTSP × N ─▶ decode ─▶ [gate] ─▶ detect ─▶ track ─▶ analytics ─▶ sink
           backend    opsional  batched   ByteTrack  counting   JSONL/SQLite/
           bisa       skip fps  fps bisa  opsional   line       webhook/stdout
           diganti              diatur
```

Setiap tahap bisa dimatikan. Sweep decode-saja adalah konfigurasi yang sah dan
merupakan pengukuran pertama yang harus jalan.

### Lapisan 2 — TETAP JALAN (ops)

Inilah yang menjawab *"misal ngejalanin program terus jebol"* dan *"observability"*.

| Komponen | Fungsi |
|---|---|
| Supervisor | satu **proses** per kamera; satu mati, 19 lain hidup |
| Restart + backoff | eksponensial + jitter, hindari reconnect storm |
| Watchdog | timestamp frame terakhir per kamera; diam > 10 s = reconnect paksa |
| Antrian berbatas | `maxsize` selalu diset, drop-oldest, drop dihitung sebagai metrik |
| Degradasi bertahap | turunkan fps deteksi sebelum menjatuhkan kamera |
| `/metrics` | format Prometheus |

**Kegagalan diam-diam lebih berbahaya daripada crash.** Stream yang berhenti mengirim
tanpa error adalah mode kegagalan nomor satu di sistem multicamera. Watchdog ada
untuk itu, bukan untuk crash.

### Lapisan 3 — TAHU BATASNYA (model)

```
sweep semua sumbu  ─▶  fit  ─▶  kapasitas mesin (Mpix/s)  ─▶  kalkulator
                                                              ─▶  prediksi mesin lain
```

Hipotesis inti: **biaya berskala dengan laju piksel (W × H × fps), bukan jumlah
stream.** Kalau benar, satu koefisien per mesin per backend sudah cukup untuk
memprediksi seluruh permukaan.

```
N_maks = kapasitas_mesin / (W × H × fps_deteksi)
```

Kalau hipotesis ini salah — misalnya ada overhead per-stream yang signifikan —
**itu temuan yang lebih menarik lagi**, dan modelnya jadi dua suku:
`biaya = a·(laju_piksel) + b·N`. Ukur `b`, jangan asumsikan nol.

---

## 4. Keputusan desain yang menyatukan proyek

### 4.1 Satu `/metrics` untuk dua tujuan

Endpoint kesehatan yang dibutuhkan untuk **mengoperasikan** sistem berisi metrik yang
persis sama dengan yang dibutuhkan untuk **mengukurnya**: fps per kamera, lag, drop,
kedalaman antrian, restart, utilisasi.

Satu implementasi melayani Grafana dan harness sekaligus. Ini yang menjaga ruang
lingkup tetap kecil sambil memenuhi seluruh gambaran sistem — dan layak dijelaskan
di README sebagai keputusan sadar, bukan kebetulan.

### 4.2 Proses, bukan thread

GIL, dan isolasi kegagalan. Satu decoder crash tidak boleh menjatuhkan yang lain.

### 4.3 Metrik utama `ingest_lag`, bukan "latency"

Publisher jalan `-re` pada fps tetap → frame ke-*k* seharusnya tiba pada `t₀ + k/fps`.

```
lag_k = t_k − (t₀ + k / fps)
```

Datar = mengimbangi. Tumbuh monoton = buffer menumpuk.

**Wajib ditulis di README:** ini **bukan** glass-to-glass. Tidak mencakup sensor,
ISP, encoder, jaringan nyata, atau buffer server RTSP. Sebutkan juga metode yang
tidak dipakai (stopwatch glass-to-glass, RTCP sender report) dan alasannya.

### 4.4 Verifikasi drop lewat nomor frame tertanam

Video uji dibangkitkan dengan indeks frame terbakar (`ffmpeg drawtext`). Konsumen
membaca ROI kecil → bisa membedakan **"kami yang membuang"** dari **"decoder diam-diam
melewatkan"**. Tanpa ini `drop_rate` tidak bisa dipercaya.

### 4.5 Warm-up, durasi tetap, catat termal

i7-1255U adalah chip 15 W. Buang **20 detik pertama**, ukur **90 detik**, catat suhu
awal dan akhir, jalankan dengan **AC terpasang**. Benchmark 10 detik tidak sah.

### 4.6 Baseline publisher-only di setiap N

Publisher dan konsumen berbagi CPU. Setiap sweep menyertakan run tanpa konsumen;
biaya harness dilaporkan terpisah. Hampir semua repo benchmark mengabaikan ini.

### 4.7 Persentil, bukan rata-rata. Timing CUDA disinkronkan

p50/p95/p99. Dan `torch.cuda.synchronize()` atau `torch.cuda.Event` di kedua sisi
timer — `perf_counter()` naif mengukur waktu antrian kernel, bukan komputasi.

---

## 5. Validasi eksternal — nilai jual utama

Tiga sumber validasi yang tidak butuh hardware tambahan:

**1. Angka terbitan NVIDIA.** Dokumentasi DeepStream memuat hasil terukur untuk Tesla
T4: H.265 mencapai 64 stream 1080p30, H.264 mencapai 39 stream, dengan ResNet10 +
tracker + 3 klasifikasi sekunder. Model Anda diuji terhadap angka itu — hardware yang
tidak akan pernah Anda pegang.

**2. Rasio codec.** 64 vs 39 di GPU yang sama berarti **H.265 ≈ 1,6× lebih hemat**.
Itu bisa direplikasi di laptop Anda dalam satu jam. Kalau rasio Anda juga sekitar
1,6×, model tervalidasi silang.

**3. Prediksi sebelum verifikasi.** Publikasikan prediksi kapasitas RTX 4060 Ti di
README **sebelum** mengujinya. Ukur nanti. Laporkan errornya, berapa pun hasilnya.
Prediksi yang dipublikasikan lebih dulu adalah metodologi sains, dan hampir tidak ada
repo GitHub yang melakukannya.

### Studi lintas-mesin

| Mesin | Biaya | Peran |
|---|---|---|
| Laptop (i7-1255U, Iris Xe, MX550) | — | mesin utama, tiga jalur decode |
| GitHub Actions runner | gratis | benchmark otomatis tiap commit |
| Kaggle / Colab T4 | gratis | **pembanding langsung ke angka NVIDIA** |
| Oracle Cloud Always Free (ARM) | gratis | arsitektur ketiga + uji 24/7 |
| RTX 4090 sewaan | ~$1 untuk 3 jam | satu hero run 20 stream sungguhan |

Kurva kapasitas di lima mesin, dengan satu model yang memprediksi kelimanya, jauh
lebih bernilai daripada satu run 20 kamera. Keterbatasan laptop bukan kelemahan —
ia salah satu titik dalam studi.

---

## 6. Yang TIDAK dibangun

| Dikecualikan | Alasan |
|---|---|
| DeepStream | tidak bisa dijalankan; bahas sebagai referensi, jangan klaim menguji |
| Perekaman / NVR | mengubah kebutuhan disk total; bukan pertanyaan kapasitas |
| ALPR / OCR | proyek lain |
| UI web sendiri | cukup ekspor JSON dashboard Grafana |
| Orkestrasi multi-mesin | jauh di luar lingkup |
| Dukungan kamera fisik | tidak punya; jangan tulis kode yang tidak bisa diuji |
| Klaim "20 kamera" tanpa kualifikasi | selalu sertakan resolusi dan fps |

---

## 7. Tangga milestone

| Ver | Isi | Selesai kalau |
|---|---|---|
| **v0.1** | rig + 1 stream + 1 reader | satu angka fps nyata ← **sebelum makan siang** |
| **v0.2** | N reader, sweep, samples.csv | tabel fps vs N |
| **v0.3** | `ingest_lag` + verifikasi drop | p50/p95/p99 vs N |
| **v0.4** | backend decode × codec | cpu/qsv/nvdec × h264/h265 ← **kirim temuan ke mentor di sini** |
| **v0.5** | tahap deteksi, fps bisa diatur | permukaan N × resolusi × fps |
| **v0.6** | fit model + **kalkulator** | pertanyaan mentor terjawab penuh |
| **v0.7** | lapisan ops: supervisor, watchdog, `/metrics`, Grafana JSON | jalan semalaman tanpa ditunggui |
| **v0.8** | analytics: counting line → event terstruktur | JSONL/SQLite keluar |
| **v1.0** | validasi lintas-mesin, README, docs, CI, rilis | orang asing bisa mereproduksi |

**Realistis 3–6 akhir pekan, bukan satu.** v0.6 sudah menjawab pertanyaan mentor;
v0.7–v1.0 yang menjadikannya portofolio.

Jangan lompat. Setiap anak tangga menghasilkan sesuatu yang bisa ditunjukkan.

---

## 8. Standar mutu

- [ ] `uv` + `pyproject.toml`, dependensi ter-pin, lockfile di-commit
- [ ] Type hints, `ruff` + `mypy` lolos
- [ ] Tes untuk yang bisa dites tanpa hardware: matematika lag, kebijakan drop, fit
      model, skema CSV. **Tes bermakna > coverage tinggi**
- [ ] GitHub Actions: lint + test, dan **benchmark CPU otomatis** — hasilnya
      di-commit sebagai salah satu titik data lintas-mesin
- [ ] Satu perintah menjalankan sweep penuh
- [ ] `runs/<id>/env.json` mencatat spesifikasi mesin otomatis
- [ ] Conventional commits, riwayat rapi
- [ ] README: hasil + plot di atas lipatan; instalasi setelahnya
- [ ] `METHODOLOGY.md`, `RESULTS.md`, `LICENSES.md`, `LICENSE`
- [ ] Tag `v1.0.0` + GitHub Release

### Narasi README

1. **Ia jalan** — pipeline multicamera, N stream → event terstruktur
2. **Ia tetap jalan** — supervisor, watchdog, metrik kesehatan
3. **Ia tahu batasnya** — model kapasitas + kalkulator
4. Divalidasi di lima mesin, termasuk terhadap angka terbitan NVIDIA
5. **Limitations** yang jujur — tulis bagian ini **sebelum** menulis Features

Setiap batasan yang Anda sebut duluan adalah keberatan yang tidak jadi diajukan.

---

## 9. Lisensi & kebersihan

| Komponen | Pilihan |
|---|---|
| Repo | Apache-2.0 |
| Detektor default | RT-DETRv2 via `transformers` (Apache-2.0) |
| Ultralytics YOLO | plugin opsional, extras group — AGPL-3.0 menular |
| Video uji | CC / public domain saja |

**Tidak boleh masuk repo, bahkan di history:** video atau frame CCTV kantor, IP
internal, hostname, path server, nama orang, nama organisasi CVAT, kredensial.
`.gitignore` (`*.mp4`, `runs/`, `.env`) di **commit pertama**.

Posisi repo: alat ukur generik. Bukan repo proyek perusahaan.

---

## 10. Risiko

| Risiko | Tanda awal | Mitigasi |
|---|---|---|
| **Membangun hal yang salah dengan rapi** | tidak ada umpan balik selama 3 minggu | kirim temuan pendek setelah v0.4 |
| Ruang lingkup melebar | menulis lapisan ops sebelum v0.4 selesai | tangga milestone, urut |
| Kode banyak, ukuran nol | jam 14 sudah 1.500 baris, belum ada CSV | v0.1 sebelum makan siang |
| Overclaim | muncul "20 kamera" tanpa resolusi | Limitations ditulis duluan |
| Hipotesis laju piksel salah | model tidak fit | ukur suku per-stream `b`, jangan asumsikan nol |
| Termal mengacaukan hasil | run ke-2 beda jauh dari run ke-1 | warm-up + durasi tetap + AC |
| Rahasia kantor bocor | video kantor dipakai uji | `.gitignore` commit pertama |

---

## 11. Umpan balik tanpa bertanya

Kita memilih tidak bertanya ke mentor. Gantinya, **tunjukkan lalu amati reaksi.**

Setelah v0.4, kirim format `Did / Do / To-do` seperti biasa — satu paragraf, satu
temuan konkret (misalnya rasio codec, atau Iris Xe vs MX550). Reaksinya akan
mengoreksi arah tanpa satu pertanyaan pun diajukan.

Itu bentuk "tidak jadi beban" yang benar: bukan diam, tapi **membawa sesuatu yang
tinggal direspons.**
