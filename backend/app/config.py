from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Konfigurasi runtime. Semua bisa di-override lewat env var berprefix JAGOJUAL_."""

    model_config = SettingsConfigDict(env_prefix="JAGOJUAL_", env_file=".env", extra="ignore")

    mode: str = "mock"  # "mock" (tanpa GPU) | "local" (LLM + adapter LoRA)
    base_model_id: str = "Qwen/Qwen2.5-7B-Instruct"
    adapter_path: str = "../model/checkpoints"
    cors_origins: str = "http://localhost:3000"

    # --- MODE=local ---
    # 4-bit wajib untuk GPU demo 6GB; matikan hanya kalau VRAM lega.
    load_4bit: bool = True
    # Default: semua layer di GPU tunggal (cepat; di 6GB masih muat tapi headroom tipis).
    # Set JAGOJUAL_CPU_OFFLOAD=1 kalau GPU-nya crash/OOM: sebagian layer pindah ke RAM (fp32)
    # supaya GPU dapat headroom, dengan konsekuensi sedikit lebih lambat. Cocok buat 6GB mepet.
    cpu_offload: bool = False
    gpu_mem_gib: int = 5   # cap VRAM saat cpu_offload aktif (GPU 8GB: naikkan ke 7)
    cpu_mem_gib: int = 12  # batas RAM untuk layer yang di-offload
    # Hanya dipakai mode Pelanggan. Mode Pelatih selalu greedy supaya penilaian
    # atas percakapan yang sama tidak berubah-ubah saat juri mengulang demo.
    temperature: float = 0.8
    max_new_tokens_chat: int = 160
    max_new_tokens_evaluate: int = 512


settings = Settings()
