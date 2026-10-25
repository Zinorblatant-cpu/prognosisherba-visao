"""Script one-shot pra calibrar câmera fixa: gera calibration.json.

Fluxo:
1. Usuário informa a região que identifica esta estação.
2. Usuário informa quantos cm mede o segmento de referência (ex: 10.0).
3. Abre preview da webcam. Usuário clica em 2 extremos do segmento.
4. Usuário clica na linha do chão (base da grama).
5. Congela frame com a grade em cm desenhada. Passar o mouse mostra a
   distância de qualquer ponto. ENTER=salvar, R=refazer, ESC=cancelar.
6. Grava calibration.json (com regiao, px_por_cm, y_chao e a grade).
"""
from __future__ import annotations

import json
import math
import os
import sys
from datetime import datetime

import cv2
import numpy as np
from dotenv import load_dotenv

import camera
import pick

load_dotenv()

# --- Configurações -----------------------------------------------------------
CAMERA_INDEX = camera.detectar_indice()
CAMERA_BACKEND = camera.detectar_backend()  # MSMF no Windows, V4L2 na Pi
OUTPUT_PATH = "calibration.json"
COR_PONTO_REF = (0, 255, 255)      # amarelo — pontos da régua
COR_LINHA_REF = (0, 200, 200)      # amarelo escuro — linha entre eles
COR_LINHA_CHAO = (255, 255, 255)   # branco — chão
COR_TEXTO = (255, 255, 255)
COR_GRADE = (170, 170, 170)        # cinza — pontos da grade em cm
JANELA = "Calibrar Camera"


# --- Funções puras (testadas) ------------------------------------------------
def calcular_px_por_cm(
    p1: tuple[int, int], p2: tuple[int, int], cm_ref: float
) -> float:
    """Distância euclidiana em pixels dividida pelo tamanho conhecido em cm."""
    if cm_ref <= 0:
        raise ValueError("cm_ref deve ser > 0")
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    dist_px = math.sqrt(dx * dx + dy * dy)
    if dist_px == 0:
        raise ValueError("pontos p1 e p2 são idênticos")
    return dist_px / cm_ref


def montar_grade(
    px_por_cm: float,
    y_chao: int,
    resolucao: tuple[int, int],
    passo_cm: float = pick.PASSO_GRADE_CM,
) -> dict:
    """Descreve a grade em cm (origem no chão) que cobre o frame.

    Com ela, cada ponto (i, j) fica a i·passo cm da borda esquerda e
    j·passo cm acima do chão. É a grade usada pelo teorema de Pick.
    """
    if passo_cm <= 0:
        raise ValueError("passo_cm deve ser > 0")
    px_por_ponto = px_por_cm * passo_cm
    return {
        "passo_cm": float(passo_cm),
        "px_por_ponto": float(px_por_ponto),
        "origem_px": [0, int(y_chao)],
        "colunas": int((resolucao[0] - 1) // px_por_ponto) + 1,
        "linhas_acima_chao": int(y_chao // px_por_ponto),
    }


def montar_calibration_dict(
    px_por_cm: float,
    y_chao: int,
    resolucao: tuple[int, int],
    cm_ref: float,
    regiao: str | None = None,
    passo_grade_cm: float = pick.PASSO_GRADE_CM,
) -> dict:
    dados = {
        "px_por_cm": float(px_por_cm),
        "y_chao": int(y_chao),
        "resolucao": [int(resolucao[0]), int(resolucao[1])],
        "created_at": datetime.now().replace(microsecond=0).isoformat(),
        "segmento_cm_referencia": float(cm_ref),
        "grade": montar_grade(px_por_cm, y_chao, resolucao, passo_grade_cm),
    }
    # Opcional: se ausente, medir_grama.py cai no REGIAO do .env como fallback.
    if regiao is not None:
        dados["regiao"] = regiao
    return dados


def salvar_calibration(dados: dict, path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dados, f, ensure_ascii=False, indent=2)


# --- UI (não testada, exige webcam) ------------------------------------------
def _ler_regiao() -> str | None:
    """Pergunta a região da estação. Retorna None se cancelado/vazio."""
    try:
        entrada = input(
            "Regiao (identifica esta estação, ex: rodoanel norte): "
        ).strip()
        if not entrada:
            print("ERRO: regiao não pode ser vazia", file=sys.stderr)
            return None
        return entrada
    except (EOFError, KeyboardInterrupt):
        return None


def _ler_cm_ref() -> float | None:
    """Pergunta ao usuário o tamanho em cm. Retorna None se cancelado/invalido."""
    try:
        entrada = input(
            "Quantos cm mede o segmento de referência (ex: 10.0)? "
        ).strip().replace(",", ".")
        valor = float(entrada)
        if valor <= 0:
            print("ERRO: valor deve ser > 0", file=sys.stderr)
            return None
        return valor
    except (ValueError, EOFError, KeyboardInterrupt):
        return None


def _abrir_webcam() -> cv2.VideoCapture:
    cap = cv2.VideoCapture(CAMERA_INDEX, CAMERA_BACKEND)
    if not cap.isOpened():
        raise RuntimeError(
            f"webcam nao acessivel (index={CAMERA_INDEX}, backend={CAMERA_BACKEND})"
        )
    return cap


def _capturar_cliques(
    cap: cv2.VideoCapture, instrucao: str, n_cliques: int
) -> list[tuple[int, int]] | None:
    """Preview ao vivo aceitando N cliques do mouse. Retorna None se ESC."""
    cliques: list[tuple[int, int]] = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(cliques) < n_cliques:
            cliques.append((x, y))

    cv2.namedWindow(JANELA)
    cv2.setMouseCallback(JANELA, on_mouse)
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            if cv2.waitKey(1) & 0xFF == 27:
                return None
            continue
        display = frame.copy()
        altura, largura = display.shape[:2]

        for frac in (0.10, 0.50, 0.90):
            y = int(altura * frac)
            cv2.line(display, (0, y), (largura, y), (200, 200, 200), 1)

        for i, (cx, cy) in enumerate(cliques):
            cv2.circle(display, (cx, cy), 6, COR_PONTO_REF, -1)
            cv2.putText(display, str(i + 1), (cx + 10, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COR_PONTO_REF, 2)
        if len(cliques) >= 2 and n_cliques >= 2:
            cv2.line(display, cliques[0], cliques[1], COR_LINHA_REF, 2)

        cv2.putText(display, instrucao, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COR_TEXTO, 1)
        cv2.putText(display, f"Cliques: {len(cliques)}/{n_cliques}",
                    (10, altura - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COR_TEXTO, 1)
        cv2.imshow(JANELA, display)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            return None
        if len(cliques) >= n_cliques:
            return cliques


def _formatar_cm(valor: float) -> str:
    return f"{valor:.1f}".replace(".", ",")


def _desenhar_grade(
    img: np.ndarray, px_por_cm: float, y_chao: int, passo_cm: float,
) -> None:
    """Pontos da grade em cm (origem no chão) + marcas a cada 5 cm.

    Pula pontos pra manter ~10px entre eles; a grade gravada continua com
    o passo cheio.
    """
    altura, largura = img.shape[:2]
    px_por_ponto = px_por_cm * passo_cm
    salto = max(1, math.ceil(10 / px_por_ponto))
    j_min = -int((altura - y_chao) // px_por_ponto)
    j_max = int(y_chao // px_por_ponto)
    n_colunas = int((largura - 1) // px_por_ponto) + 1
    for j in range(j_min - j_min % salto, j_max + 1, salto):
        for i in range(0, n_colunas, salto):
            x, y = pick.ponto_grade_para_pixel(i, j, px_por_cm, y_chao, passo_cm)
            if 0 <= y < altura:
                img[y, x] = COR_GRADE
    # Marcas numeradas a cada 5 cm acima do chão, na borda direita.
    for cm in range(5, int(y_chao / px_por_cm) + 1, 5):
        y = round(y_chao - cm * px_por_cm)
        cv2.line(img, (largura - 15, y), (largura, y), COR_GRADE, 1)
        cv2.putText(img, f"{cm}cm", (largura - 55, y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, COR_GRADE, 1)


def _desenhar_leitura_cursor(
    img: np.ndarray, cursor: tuple[int, int], px_por_cm: float,
    y_chao: int, passo_cm: float,
) -> None:
    """Encaixa o cursor no ponto da grade mais próximo e mostra a distância."""
    px_por_ponto = px_por_cm * passo_cm
    i = round(cursor[0] / px_por_ponto)
    j = round((y_chao - cursor[1]) / px_por_ponto)
    x, y = pick.ponto_grade_para_pixel(i, j, px_por_cm, y_chao, passo_cm)
    x_cm, altura_cm = pick.pixel_para_cm(x, y, px_por_cm, y_chao)
    sinal = "+" if altura_cm >= 0 else "-"
    texto = (f"x={_formatar_cm(x_cm)}cm  "
             f"altura={sinal}{_formatar_cm(abs(altura_cm))}cm")
    cv2.circle(img, (x, y), 4, COR_TEXTO, 1)
    largura = img.shape[1]
    (tw, _), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    tx = min(x + 10, largura - tw - 5)
    ty = max(y - 10, 15)
    cv2.putText(img, texto, (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COR_TEXTO, 1)


def _confirmar(
    cap: cv2.VideoCapture, p1: tuple[int, int], p2: tuple[int, int],
    y_chao: int, px_por_cm: float, cm_ref: float,
) -> str:
    """Congela um frame com marcações e a grade em cm, aguarda ENTER/R/ESC.

    Passar o mouse mostra a distância do ponto da grade mais próximo.
    """
    ok, frame = cap.read()
    if not ok or frame is None:
        frame = np.zeros((480, 640, 3), np.uint8)
    display = frame.copy()
    altura, largura = display.shape[:2]

    _desenhar_grade(display, px_por_cm, y_chao, pick.PASSO_GRADE_CM)
    cv2.line(display, (0, y_chao), (largura, y_chao), COR_LINHA_CHAO, 2)
    cv2.putText(display, "chao", (5, y_chao - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COR_LINHA_CHAO, 1)

    cv2.line(display, p1, p2, COR_LINHA_REF, 2)
    for ponto in (p1, p2):
        cv2.circle(display, ponto, 6, COR_PONTO_REF, -1)

    linha1 = f"{cm_ref:g}cm = {math.hypot(p2[0]-p1[0], p2[1]-p1[1]):.1f}px"
    linha2 = f"→ {px_por_cm:.2f} px/cm"
    cv2.putText(display, linha1, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COR_TEXTO, 1)
    cv2.putText(display, linha2, (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COR_TEXTO, 1)
    cv2.putText(display, "mouse=distancia  ENTER=salvar  R=refazer  ESC=cancelar",
                (10, altura - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COR_TEXTO, 1)

    cursor: list[tuple[int, int]] = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            cursor[:] = [(x, y)]

    cv2.setMouseCallback(JANELA, on_mouse)
    while True:
        tela = display.copy()
        if cursor:
            _desenhar_leitura_cursor(tela, cursor[0], px_por_cm, y_chao,
                                     pick.PASSO_GRADE_CM)
        cv2.imshow(JANELA, tela)
        key = cv2.waitKey(30) & 0xFF
        if key == 27:
            return "cancelar"
        if key == 13:  # ENTER
            return "salvar"
        if key in (ord("r"), ord("R")):
            return "refazer"


def main() -> int:
    regiao = _ler_regiao()
    if regiao is None:
        print("Cancelado.")
        return 130

    cm_ref = _ler_cm_ref()
    if cm_ref is None:
        print("Cancelado.")
        return 130

    try:
        cap = _abrir_webcam()
    except RuntimeError as e:
        print(f"ERRO: {e}", file=sys.stderr)
        return 1

    try:
        while True:
            print(
                f"\nPreview aberto. Clique nos 2 extremos do segmento de "
                f"{cm_ref:g}cm.",
                flush=True,
            )
            pontos_ref = _capturar_cliques(
                cap, f"Clique nos 2 extremos do segmento de {cm_ref:g}cm",
                n_cliques=2,
            )
            if pontos_ref is None:
                print("Cancelado.")
                return 130

            print("Agora clique na linha do chao (base da grama).", flush=True)
            pontos_chao = _capturar_cliques(
                cap, "Clique na linha do chao (base da grama)",
                n_cliques=1,
            )
            if pontos_chao is None:
                print("Cancelado.")
                return 130

            p1, p2 = pontos_ref[0], pontos_ref[1]
            y_chao = pontos_chao[0][1]
            try:
                px_por_cm = calcular_px_por_cm(p1, p2, cm_ref)
            except ValueError as e:
                print(f"ERRO: {e}. Refazendo...", file=sys.stderr)
                continue

            decisao = _confirmar(cap, p1, p2, y_chao, px_por_cm, cm_ref)
            if decisao == "cancelar":
                print("Cancelado.")
                return 130
            if decisao == "refazer":
                continue

            resolucao = (
                int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            )
            dados = montar_calibration_dict(
                px_por_cm, y_chao, resolucao, cm_ref, regiao,
            )
            salvar_calibration(dados, OUTPUT_PATH)
            grade = dados["grade"]
            print(
                f"\nOK: {OUTPUT_PATH} salvo.\n"
                f"   regiao    = {regiao}\n"
                f"   px_por_cm = {px_por_cm:.2f}\n"
                f"   y_chao    = {y_chao}\n"
                f"   resolucao = {resolucao[0]}x{resolucao[1]}\n"
                f"   grade     = 1 ponto a cada {_formatar_cm(grade['passo_cm'])}"
                f" cm ({grade['colunas']} colunas x "
                f"{grade['linhas_acima_chao']} linhas acima do chao)"
            )
            for nome, ponto in (("P1", p1), ("P2", p2)):
                x_cm, altura_cm = pick.pixel_para_cm(*ponto, px_por_cm, y_chao)
                print(f"   {nome}        = x {_formatar_cm(x_cm)} cm, "
                      f"{_formatar_cm(altura_cm)} cm acima do chao")
            return 0
    except KeyboardInterrupt:
        print("Cancelado.")
        return 130
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())
