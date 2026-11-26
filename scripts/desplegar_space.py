"""Construye y publica el Space de Hugging Face (Gradio + ZeroGPU).

Uso::

    python scripts/desplegar_space.py --repo Jeol28/voley-ar            # crea/actualiza y sube
    python scripts/desplegar_space.py --solo-construir                  # solo arma dist/space

Requiere haber iniciado sesión (``hf auth login``). El Space se crea con
hardware ZeroGPU (``zero-a10g``); una cuenta gratuita puede tener hasta 2.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
DESTINO = RAIZ / "dist" / "space"
INCLUIR = ["app.py", "LICENSE", "voleyar", "scripts/generar_imprimibles.py", "scripts/calibrar_camara.py",
           "docs/imprimibles", "docs/capturas", "docs/arquitectura.md", "docs/guion_video.md"]

NOTA_SPACE = """
> **Versión en línea (Hugging Face Space con ZeroGPU).** Imprime la cancha (`docs/imprimibles/cancha.pdf`), calibra
> tu cámara en la pestaña **Calibración** y juega en **Jugar → En el navegador**. La calibración y la piel son de tu
> sesión: no se guardan en el servidor (descarga `camara.json` para usarlo en tu equipo). Depth Anything y OWLv2
> piden la GPU solo cuando se usan. La versión de escritorio (`python jugar.py`) es más fluida.
"""


def construir() -> Path:
    if DESTINO.exists():
        shutil.rmtree(DESTINO)
    DESTINO.mkdir(parents=True)
    for rel in INCLUIR:
        origen = RAIZ / rel
        destino = DESTINO / rel
        destino.parent.mkdir(parents=True, exist_ok=True)
        if origen.is_dir():
            shutil.copytree(origen, destino, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"))
        else:
            shutil.copy2(origen, destino)
    shutil.copy2(RAIZ / "despliegue" / "requirements.txt", DESTINO / "requirements.txt")
    encabezado = (RAIZ / "despliegue" / "encabezado_readme.yml").read_text(encoding="utf-8")
    readme = (RAIZ / "README.md").read_text(encoding="utf-8")
    titulo, resto = readme.split("\n", 1)
    (DESTINO / "README.md").write_text(encabezado + "\n" + titulo + "\n" + NOTA_SPACE + resto, encoding="utf-8")
    return DESTINO


def publicar(repo: str, privado: bool, hardware: str) -> str:
    from huggingface_hub import HfApi

    api = HfApi()
    if api.repo_exists(repo, repo_type="space"):
        # Reemplaza el contenido en un solo commit: borra lo que no esté en dist/space (salvo .gitattributes,
        # que guarda las reglas de LFS/Xet de los binarios). El historial del Space conserva lo anterior.
        nuevos = {p.relative_to(DESTINO).as_posix() for p in DESTINO.rglob("*") if p.is_file()}
        viejos = set(api.list_repo_files(repo, repo_type="space"))
        borrar = sorted(viejos - nuevos - {".gitattributes"})
    else:
        api.create_repo(repo, repo_type="space", space_sdk="gradio", private=privado, space_hardware=hardware)
        borrar = []
    api.upload_folder(folder_path=str(DESTINO), repo_id=repo, repo_type="space",
                      commit_message="Despliegue de Voley AR", delete_patterns=borrar or None)
    if api.get_space_runtime(repo).requested_hardware != hardware:
        api.request_space_hardware(repo, hardware)
    return f"https://huggingface.co/spaces/{repo}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default="Jeol28/voley-ar")
    ap.add_argument("--privado", action="store_true")
    ap.add_argument("--hardware", default="zero-a10g")
    ap.add_argument("--solo-construir", action="store_true")
    a = ap.parse_args()
    ruta = construir()
    print(f"Space armado en {ruta.relative_to(RAIZ)}")
    if not a.solo_construir:
        print("Publicado:", publicar(a.repo, a.privado, a.hardware))


if __name__ == "__main__":
    sys.exit(main())
