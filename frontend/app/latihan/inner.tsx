"use client";

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { useEffect, useRef, useState } from "react";

import ModeBadge from "../../components/ModeBadge";
import {
  ChatMessage,
  evaluate,
  EvaluateResponse,
  getScenario,
  Scenario,
  sendChat,
  TeknikScore,
} from "../../lib/api";

/** Nama tampilan teknik. Sumber kebenaran label tetap `data/taxonomy.json`. */
const NAMA_TEKNIK: Record<string, string> = {
  sapa_rapport: "Sapaan & rapport",
  gali_kebutuhan: "Menggali kebutuhan",
  presentasi_manfaat: "Presentasi manfaat",
  atasi_keberatan: "Mengatasi keberatan",
  closing: "Closing",
  upsell: "Upsell / cross-sell",
};

function nadaSkor(skor: number) {
  if (skor >= 80) return { teks: "text-kuat", bg: "bg-kuat" };
  if (skor >= 60) return { teks: "text-sedang", bg: "bg-sedang" };
  return { teks: "text-lemah", bg: "bg-lemah" };
}

/** Satu kalimat yang membuat angka total berarti tanpa perlu ditafsirkan sendiri. */
function putusan(skor: number): string {
  if (skor >= 85) return "Tekniknya sudah kuat dan konsisten.";
  if (skor >= 70) return "Arahnya sudah benar, tinggal dirapikan.";
  if (skor >= 55) return "Beberapa teknik inti masih terlewat.";
  return "Banyak momen penting yang belum tergarap.";
}

export default function LatihanInner() {
  const id = useSearchParams().get("id") || "";
  const [scenario, setScenario] = useState<Scenario | null>(null);
  const [history, setHistory] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [result, setResult] = useState<EvaluateResponse | null>(null);
  const akhirPercakapan = useRef<HTMLDivElement>(null);

  // --- Voice (Web Speech API — berjalan di browser, tanpa server/GPU) ---
  // Fitur opsional; mode teks tetap jadi jalur utama. STT & TTS memakai bahasa id-ID.
  const [voiceOn, setVoiceOn] = useState(false); // baca balasan pelanggan dengan suara
  const [listening, setListening] = useState(false); // sedang merekam suara sales
  const recognitionRef = useRef<SpeechRecognitionLike | null>(null);
  const terakhirDibaca = useRef(0);

  useEffect(() => {
    if (!id) return;
    getScenario(id)
      .then((s) => {
        setScenario(s);
        setHistory([{ role: "pelanggan", text: s.pembuka }]);
      })
      .catch((e) => setErr(e?.message ?? "Gagal memuat skenario."));
  }, [id]);

  // Balasan pelanggan bisa panjang; tanpa ini giliran terbaru jatuh di bawah layar.
  useEffect(() => {
    akhirPercakapan.current?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }, [history, loading]);

  // Bacakan giliran pelanggan terbaru saat mode suara aktif.
  useEffect(() => {
    if (!voiceOn) return;
    const terakhir = history[history.length - 1];
    if (terakhir?.role === "pelanggan" && history.length > terakhirDibaca.current) {
      terakhirDibaca.current = history.length;
      bicara(terakhir.text);
    }
  }, [history, voiceOn]);

  // Hentikan suara & rekaman kalau komponen dilepas / pindah halaman.
  useEffect(() => {
    return () => {
      recognitionRef.current?.stop();
      if (typeof window !== "undefined") window.speechSynthesis?.cancel();
    };
  }, []);

  function bicara(teks: string) {
    if (typeof window === "undefined" || !("speechSynthesis" in window)) return;
    window.speechSynthesis.cancel();
    const ucap = new SpeechSynthesisUtterance(teks);
    ucap.lang = "id-ID";
    window.speechSynthesis.speak(ucap);
  }

  function toggleDengar() {
    const SR =
      (window as unknown as { SpeechRecognition?: SpeechRecognitionCtor; webkitSpeechRecognition?: SpeechRecognitionCtor })
        .SpeechRecognition ||
      (window as unknown as { webkitSpeechRecognition?: SpeechRecognitionCtor }).webkitSpeechRecognition;
    if (!SR) {
      setErr("Browser ini belum mendukung input suara. Coba pakai Chrome atau Edge.");
      return;
    }
    if (listening) {
      recognitionRef.current?.stop();
      return;
    }
    window.speechSynthesis?.cancel(); // jangan sampai TTS & mic bunyi barengan
    const rec = new SR();
    rec.lang = "id-ID";
    rec.interimResults = false;
    rec.continuous = false;
    rec.onresult = (e: SpeechRecognitionEventLike) => {
      const teks = e.results[0][0].transcript;
      setInput((prev) => (prev ? prev + " " : "") + teks);
    };
    rec.onend = () => setListening(false);
    rec.onerror = () => setListening(false);
    recognitionRef.current = rec;
    setListening(true);
    rec.start();
  }

  async function kirim() {
    if (!input.trim() || !scenario || loading) return;
    const salesMsg: ChatMessage = { role: "sales", text: input.trim() };
    const next = [...history, salesMsg];
    setHistory(next);
    setInput("");
    setLoading(true);
    setErr(null);
    try {
      const { reply } = await sendChat(scenario.id, salesMsg.text, next);
      setHistory((h) => [...h, { role: "pelanggan", text: reply }]);
    } catch (e: unknown) {
      // Backend menjelaskan sebabnya (mis. 503 saat model lokal gagal dimuat).
      // Tanpa ditampilkan, layar cuma diam dan pengguna mengira aplikasinya hang.
      setErr(e instanceof Error ? e.message : "Gagal mengirim pesan.");
    } finally {
      setLoading(false);
    }
  }

  async function selesai() {
    if (!scenario || loading) return;
    setLoading(true);
    setErr(null);
    try {
      setResult(await evaluate(scenario.id, history));
    } catch (e: unknown) {
      setErr(e instanceof Error ? e.message : "Gagal menilai percakapan.");
    } finally {
      setLoading(false);
    }
  }

  if (!scenario) {
    return (
      <main className="mx-auto max-w-2xl px-6 py-16">
        {err ? <Galat teks={err} /> : <p className="text-tinta-samar">Memuat skenario…</p>}
        <Link href="/" className="mt-6 inline-block text-sm text-tinta-lembut underline-offset-4 hover:underline">
          ← Kembali ke daftar
        </Link>
      </main>
    );
  }

  // ---------------------------------------------------------------- rapor ---
  if (result) {
    const nada = nadaSkor(result.skor_total);
    return (
      <main className="mx-auto max-w-2xl px-6 py-12 sm:py-16">
        <p className="eyebrow">Rapor latihan</p>
        <h1 className="mt-2 font-display text-3xl font-bold tracking-tight">{scenario.judul}</h1>

        <div className="mt-8 flex items-end gap-5 border-b border-garis pb-8">
          <div className="font-display text-7xl font-bold leading-none tabular-nums">
            {result.skor_total}
            <span className="ml-1 align-top text-2xl font-normal text-tinta-samar">/100</span>
          </div>
          <p className={`pb-2 text-[0.975rem] font-medium leading-snug ${nada.teks}`}>
            {putusan(result.skor_total)}
          </p>
        </div>

        <section className="mt-8">
          <h2 className="eyebrow">Per teknik</h2>
          <div className="mt-4 divide-y divide-garis border-y border-garis">
            {result.per_teknik.map((t) => (
              <BarisTeknik key={t.teknik} t={t} />
            ))}
          </div>
        </section>

        {result.saran.length > 0 && (
          <section className="mt-10">
            <h2 className="eyebrow">Yang perlu dibenahi</h2>
            <ol className="mt-4 space-y-4">
              {result.saran.map((s, i) => (
                <li key={i} className="flex gap-4">
                  <span className="font-display text-xl font-bold leading-none text-bata tabular-nums">
                    {i + 1}
                  </span>
                  <p className="text-[0.925rem] leading-relaxed text-tinta-lembut">{s}</p>
                </li>
              ))}
            </ol>
          </section>
        )}

        <div className="mt-12 flex flex-wrap gap-3">
          {/* Tombol, bukan Link: URL-nya sama persis sehingga navigasi tidak akan
              memasang ulang komponen — cukup setel ulang state percakapannya. */}
          <button
            onClick={() => {
              setResult(null);
              setErr(null);
              terakhirDibaca.current = 0;
              setHistory([{ role: "pelanggan", text: scenario.pembuka }]);
            }}
            className="bg-tinta px-5 py-2.5 text-sm font-medium text-kertas transition hover:bg-bata-tua"
          >
            Ulangi skenario ini
          </button>
          <Link
            href="/"
            className="border border-garis-tua px-5 py-2.5 text-sm font-medium transition hover:border-tinta"
          >
            Pilih skenario lain
          </Link>
        </div>
      </main>
    );
  }

  // ------------------------------------------------------------ percakapan ---
  const adaRespons = history.some((m) => m.role === "sales");

  return (
    <main className="mx-auto flex min-h-screen max-w-2xl flex-col px-6 py-10">
      <header className="border-b border-garis pb-5">
        <div className="flex items-center justify-between gap-4">
          <Link href="/" className="text-sm text-tinta-lembut underline-offset-4 hover:underline">
            ← Ganti skenario
          </Link>
          <div className="flex items-center gap-3">
            <button
              onClick={() => setVoiceOn((v) => !v)}
              aria-pressed={voiceOn}
              title="Bacakan balasan pelanggan dengan suara"
              className={`border px-2.5 py-1 text-xs font-medium transition ${
                voiceOn ? "border-bata bg-bata-muda/40 text-bata-tua" : "border-garis-tua text-tinta-lembut hover:border-tinta"
              }`}
            >
              {voiceOn ? "🔊 Suara: on" : "🔇 Suara: off"}
            </button>
            <ModeBadge />
          </div>
        </div>
        <h1 className="mt-3 font-display text-2xl font-bold tracking-tight">{scenario.judul}</h1>
        <p className="mt-1 text-sm text-tinta-samar">
          {scenario.produk} · pelanggan {scenario.persona.tipe}
        </p>
        <p className="mt-2 text-sm leading-relaxed text-tinta-lembut">{scenario.persona.deskripsi}</p>
      </header>

      <div className="flex-1 space-y-5 py-7">
        {history.map((m, i) => (
          <Gelembung key={i} pesan={m} />
        ))}
        {loading && (
          <p className="text-sm text-tinta-samar">
            <span className="eyebrow">Pelanggan</span> sedang menimbang…
          </p>
        )}
        <div ref={akhirPercakapan} />
      </div>

      {err && (
        <div className="mb-4">
          <Galat teks={err} />
        </div>
      )}

      <div className="sticky bottom-0 -mx-6 border-t border-garis bg-kertas/95 px-6 pb-6 pt-4 backdrop-blur">
        <div className="flex gap-2">
          <button
            onClick={toggleDengar}
            title={listening ? "Berhenti merekam" : "Bicara (ubah suara jadi teks)"}
            aria-label="Input suara"
            className={`shrink-0 border px-3 py-2.5 text-sm transition ${
              listening
                ? "animate-pulse border-bata bg-bata text-kertas"
                : "border-garis-tua hover:border-bata"
            }`}
          >
            {listening ? "● Merekam…" : "🎤"}
          </button>
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && kirim()}
            placeholder="Tulis atau ucapkan jawabanmu sebagai sales…"
            aria-label="Respons jualan"
            className="min-w-0 flex-1 border border-garis-tua bg-white px-4 py-2.5 text-sm outline-none transition placeholder:text-tinta-samar focus:border-bata"
          />
          <button
            onClick={kirim}
            disabled={loading || !input.trim()}
            className="shrink-0 bg-tinta px-5 py-2.5 text-sm font-medium text-kertas transition hover:bg-bata-tua disabled:cursor-not-allowed disabled:opacity-40"
          >
            Kirim
          </button>
        </div>
        <button
          onClick={selesai}
          disabled={loading || !adaRespons}
          className="mt-2 w-full border border-garis-tua py-2.5 text-sm font-medium transition hover:border-tinta disabled:cursor-not-allowed disabled:opacity-40"
        >
          {adaRespons ? "Selesai & nilai percakapan" : "Jawab dulu untuk bisa dinilai"}
        </button>
      </div>
    </main>
  );
}

// Tipe minimal Web Speech API (belum ada di lib TS standar).
type SpeechRecognitionEventLike = { results: { [i: number]: { [j: number]: { transcript: string } } } };
interface SpeechRecognitionLike {
  lang: string;
  interimResults: boolean;
  continuous: boolean;
  onresult: (e: SpeechRecognitionEventLike) => void;
  onend: () => void;
  onerror: () => void;
  start: () => void;
  stop: () => void;
}
type SpeechRecognitionCtor = new () => SpeechRecognitionLike;

function Gelembung({ pesan }: { pesan: ChatMessage }) {
  const sales = pesan.role === "sales";
  return (
    <div className={sales ? "flex flex-col items-end" : "flex flex-col items-start"}>
      <span className="eyebrow mb-1.5">{sales ? "Kamu" : "Pelanggan"}</span>
      <p
        className={`max-w-[85%] whitespace-pre-wrap px-4 py-3 text-[0.925rem] leading-relaxed ${
          sales ? "bg-tinta text-kertas" : "border border-garis bg-white"
        }`}
      >
        {pesan.text}
      </p>
    </div>
  );
}

function BarisTeknik({ t }: { t: TeknikScore }) {
  const nada = nadaSkor(t.skor);
  const lebar = Math.max(0, Math.min(100, t.skor));
  return (
    <div className="py-4">
      <div className="flex items-baseline justify-between gap-4">
        <span className="text-sm font-medium">{NAMA_TEKNIK[t.teknik] ?? t.teknik.replace(/_/g, " ")}</span>
        <span className={`text-sm font-semibold tabular-nums ${nada.teks}`}>{t.skor}</span>
      </div>
      <div className="mt-2 h-1 bg-kertas-tua">
        <div className={`h-1 ${nada.bg}`} style={{ width: `${lebar}%` }} />
      </div>
      {/* Catatan inilah umpan balik pelatihnya — angka saja tidak mengajari apa pun. */}
      {t.catatan && <p className="mt-2 text-[0.8rem] leading-relaxed text-tinta-samar">{t.catatan}</p>}
    </div>
  );
}

function Galat({ teks }: { teks: string }) {
  return (
    <p role="alert" className="border-l-2 border-lemah bg-bata-muda/40 px-4 py-3 text-sm text-bata-tua">
      {teks}
    </p>
  );
}
