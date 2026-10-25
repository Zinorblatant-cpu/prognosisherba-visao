"""Testes da grade em cm e do teorema de Pick."""
import pytest

import pick


def _shoelace(vertices):
    n = len(vertices)
    dobro = sum(
        vertices[k][0] * vertices[(k + 1) % n][1]
        - vertices[(k + 1) % n][0] * vertices[k][1]
        for k in range(n)
    )
    return abs(dobro) / 2


def _trapezios(alturas):
    return sum((a + b) / 2 for a, b in zip(alturas, alturas[1:]))


# --- Pick ---------------------------------------------------------------------
def test_area_pick_retangulo():
    # 4x3: B = 14, I = 3*2 = 6 → 6 + 7 - 1 = 12
    ret = [(0, 0), (4, 0), (4, 3), (0, 3)]
    assert pick.pontos_na_borda(ret) == 14
    assert pick.pontos_interiores(ret) == 6
    assert pick.area_pick(ret) == 12


def test_pontos_na_borda_diagonal_usa_gcd():
    # Aresta (0,0)→(6,4): gcd(6,4) = 2 pontos (sem contar o final)
    tri = [(0, 0), (6, 4), (6, 0)]
    assert pick.pontos_na_borda(tri) == 2 + 4 + 6


@pytest.mark.parametrize("vertices", [
    [(0, 0), (6, 4), (6, 0)],                          # triângulo
    [(0, 0), (5, 0), (5, 5), (3, 2), (0, 5)],          # côncavo
    [(1, 1), (7, 2), (9, 8), (4, 11), (0, 6)],         # irregular
    [(0, 0), (0, 4), (4, 4), (4, 0)],                  # sentido horário
])
def test_area_pick_igual_shoelace(vertices):
    assert pick.area_pick(vertices) == pytest.approx(_shoelace(vertices))


def test_area_pick_degenerado_da_zero():
    assert pick.area_pick([(0, 0), (2, 0), (4, 0)]) == 0.0
    assert pick.area_pick([(0, 0), (0, 0), (3, 3)]) == 0.0


@pytest.mark.parametrize("alturas", [
    [3, 5, 2, 7, 4],
    [0, 4, 4, 0, 0, 6, 1, 0],   # zeros no meio e nas bordas
    [5, 0, 5],                   # dois trechos separados por um zero
    [0, 0, 0],
    [2, 9],
])
def test_area_perfil_pick_igual_trapezios(alturas):
    assert pick.area_perfil_pick(alturas) == pytest.approx(_trapezios(alturas))


def test_pick_perfil_conta_pontos_e_poligonos():
    # Retângulo 2x3 (colunas 0..2, altura 3): I = 1*2 = 2, B = 10 → 2 + 5 - 1 = 6
    r = pick.pick_perfil([3, 3, 3])
    assert (r["interiores"], r["borda"], r["poligonos"]) == (2, 10, 1)
    assert r["area"] == 6
    # Dois tufos separados por um zero: dois polígonos
    assert pick.pick_perfil([2, 0, 2])["poligonos"] == 2


def test_pick_perfil_sem_grama():
    assert pick.pick_perfil([0, 0]) == {
        "area": 0, "interiores": 0, "borda": 0, "poligonos": 0,
    }


def test_altura_media_cm_perfil_plano():
    # 10 pontos de altura com passo 0,5 cm → 5 cm
    assert pick.altura_media_cm([10] * 6, 0.5) == pytest.approx(5.0)


def test_altura_media_cm_exige_duas_colunas():
    with pytest.raises(ValueError, match="2 colunas"):
        pick.altura_media_cm([4], 0.5)


# --- Grade ----------------------------------------------------------------------
def test_pixel_para_cm():
    # 8 px/cm, chão em y=400: pixel (80, 320) → 10 cm da borda, 10 cm de altura
    assert pick.pixel_para_cm(80, 320, 8.0, 400) == (10.0, 10.0)
    _, abaixo = pick.pixel_para_cm(0, 408, 8.0, 400)
    assert abaixo == -1.0


def test_ponto_grade_para_pixel_ida_e_volta():
    x, y = pick.ponto_grade_para_pixel(5, 6, 8.0, 400, 0.5)
    assert (x, y) == (20, 376)
    assert pick.pixel_para_cm(x, y, 8.0, 400) == (2.5, 3.0)


def test_altura_em_pontos():
    assert pick.altura_em_pontos(None, 400, 4.0) == 0
    assert pick.altura_em_pontos(410, 400, 4.0) == 0
    assert pick.altura_em_pontos(400, 400, 4.0) == 0
    assert pick.altura_em_pontos(360, 400, 4.0) == 10
    assert pick.altura_em_pontos(359, 400, 4.0) == 10  # 10,25 → 10
