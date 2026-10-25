"""Grade calibrada em cm + teorema de Pick.

A calibração define uma grade de pontos com passo fixo em cm. Origem no
chão: x=0 na borda esquerda do frame, y=0 na linha do chão, y cresce pra
cima. Assim todo pixel tem uma distância conhecida em cm.

Teorema de Pick: num polígono simples com vértices em pontos da grade,
    A = I + B/2 - 1
I = pontos da grade estritamente dentro, B = pontos sobre a borda.
A sai em "unidades de grade²"; multiplicando por passo_cm² vira cm².
"""
from __future__ import annotations

import math

import numpy as np

PASSO_GRADE_CM = 0.5  # 1 ponto da grade a cada 0,5 cm


# --- Grade --------------------------------------------------------------------
def pixel_para_cm(
    x: int, y: int, px_por_cm: float, y_chao: int
) -> tuple[float, float]:
    """Pixel → (distância da borda esquerda, altura acima do chão), em cm.

    Altura negativa = ponto abaixo do chão calibrado.
    """
    return x / px_por_cm, (y_chao - y) / px_por_cm


def ponto_grade_para_pixel(
    i: int, j: int, px_por_cm: float, y_chao: int, passo_cm: float
) -> tuple[int, int]:
    """Ponto (i, j) da grade → pixel (x, y)."""
    px_por_ponto = px_por_cm * passo_cm
    return round(i * px_por_ponto), round(y_chao - j * px_por_ponto)


def altura_em_pontos(
    y_topo: int | None, y_chao: int, px_por_ponto: float
) -> int:
    """Topo do verde (pixel) → altura em pontos da grade, arredondada.

    None (sem verde) ou topo no chão/abaixo → 0.
    """
    if y_topo is None or y_topo >= y_chao:
        return 0
    return round((y_chao - y_topo) / px_por_ponto)


# --- Pick ---------------------------------------------------------------------
def _limpar_vertices(
    vertices: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Remove vértices repetidos em sequência (inclusive o fechamento)."""
    limpos: list[tuple[int, int]] = []
    for v in vertices:
        if not limpos or limpos[-1] != v:
            limpos.append(v)
    if len(limpos) > 1 and limpos[0] == limpos[-1]:
        limpos.pop()
    return limpos


def pontos_na_borda(vertices: list[tuple[int, int]]) -> int:
    """B: cada aresta (dx, dy) passa por gcd(|dx|, |dy|) pontos da grade."""
    n = len(vertices)
    total = 0
    for k in range(n):
        x1, y1 = vertices[k]
        x2, y2 = vertices[(k + 1) % n]
        total += math.gcd(abs(x2 - x1), abs(y2 - y1))
    return total


def pontos_interiores(vertices: list[tuple[int, int]]) -> int:
    """I: conta os pontos da grade estritamente dentro do polígono.

    Ray casting vetorizado, com aritmética inteira exata (sem float) e
    descartando os pontos que caem sobre a borda.
    """
    xs = [v[0] for v in vertices]
    ys = [v[1] for v in vertices]
    # Pontos interiores nunca ficam na borda da caixa envolvente.
    gx = np.arange(min(xs) + 1, max(xs), dtype=np.int64)
    gy = np.arange(min(ys) + 1, max(ys), dtype=np.int64)
    if gx.size == 0 or gy.size == 0:
        return 0
    px, py = np.meshgrid(gx, gy)
    dentro = np.zeros(px.shape, dtype=bool)
    na_borda = np.zeros(px.shape, dtype=bool)
    n = len(vertices)
    for k in range(n):
        x1, y1 = vertices[k]
        x2, y2 = vertices[(k + 1) % n]
        dx, dy = x2 - x1, y2 - y1
        colinear = (px - x1) * dy - (py - y1) * dx == 0
        no_segmento = (
            (px >= min(x1, x2)) & (px <= max(x1, x2))
            & (py >= min(y1, y2)) & (py <= max(y1, y2))
        )
        na_borda |= colinear & no_segmento
        if dy == 0:
            continue
        cruza = (y1 > py) != (y2 > py)
        # px < x da interseção, multiplicado por dy (sinal decide o lado).
        lhs = (px - x1) * dy
        rhs = (py - y1) * dx
        a_esquerda = lhs < rhs if dy > 0 else lhs > rhs
        dentro ^= cruza & a_esquerda
    return int(np.count_nonzero(dentro & ~na_borda))


def area_pick(vertices: list[tuple[int, int]]) -> float:
    """Área (unidades de grade²) de um polígono simples via A = I + B/2 - 1.

    Polígono degenerado (todos os vértices na mesma reta) → 0.0.
    """
    v = _limpar_vertices(vertices)
    if len(v) < 3:
        return 0.0
    # Só pra detectar polígono achatado, onde Pick não vale.
    dobro_area = sum(
        v[k][0] * v[(k + 1) % len(v)][1] - v[(k + 1) % len(v)][0] * v[k][1]
        for k in range(len(v))
    )
    if dobro_area == 0:
        return 0.0
    return pontos_interiores(v) + pontos_na_borda(v) / 2 - 1


def area_perfil_pick(alturas: list[int]) -> float:
    """Área sob um perfil de alturas em colunas consecutivas da grade.

    Cada trecho contínuo com altura > 0 vira um polígono separado, fechado
    no chão nas colunas vizinhas com altura 0. Assim nenhum polígono encosta
    em si mesmo no chão (Pick exige polígono simples).
    """
    total = 0.0
    n = len(alturas)
    k = 0
    while k < n:
        if alturas[k] <= 0:
            k += 1
            continue
        ini = k
        while k < n and alturas[k] > 0:
            k += 1
        fim = k - 1
        x_esq = ini - 1 if ini > 0 else ini
        x_dir = fim + 1 if fim < n - 1 else fim
        vertices = [(x_esq, 0)]
        vertices.extend((x, alturas[x]) for x in range(ini, fim + 1))
        vertices.append((x_dir, 0))
        total += area_pick(vertices)
    return total


def altura_media_cm(alturas: list[int], passo_cm: float) -> float:
    """Altura média (cm) de uma faixa: área do perfil ÷ largura da faixa.

    área cm² = área_grade · passo², largura cm = colunas · passo.
    """
    largura = len(alturas) - 1
    if largura < 1:
        raise ValueError("faixa precisa de pelo menos 2 colunas da grade")
    return area_perfil_pick(alturas) / largura * passo_cm
