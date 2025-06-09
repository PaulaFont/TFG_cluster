import unittest
from graph_utils import are_strings_similar

class TestSimilitudCadenas(unittest.TestCase):
    # Nombres de personas
    def test_nombres_identicos(self):
        self.assertTrue(are_strings_similar("María López", "María López"))

    def test_nombres_con_tildes(self):
        self.assertTrue(are_strings_similar("Jose Alvarez", "José Álvarez"))

    def test_nombres_con_typo(self):
        self.assertTrue(are_strings_similar("Antonio García", "Antnio Garcia"))

    def test_nombres_distintos(self):
        self.assertFalse(are_strings_similar("Lucía Gómez", "Pedro Martínez"))

    # Organizaciones
    def test_siglas_identicas(self):
        self.assertTrue(are_strings_similar("CNT", "CNT"))

    def test_siglas_con_puntos(self):
        self.assertTrue(are_strings_similar("CNT", "C.N.T."))

    def test_siglas_parecidas(self):
        self.assertTrue(are_strings_similar("UGT", "U.G.T."))

    def test_siglas_diferentes(self):
        self.assertFalse(are_strings_similar("CNT", "UGT"))

    # Fechas y años
    def test_anios_identicos(self):
        self.assertTrue(are_strings_similar("1936", "1936"))

    def test_anios_con_espacios(self):
        self.assertTrue(are_strings_similar("1936", " 1936 "))

    def test_fechas_similares(self):
        self.assertTrue(are_strings_similar("12 de julio de 1936", "12 julio 1936"))

    def test_fechas_distintas(self):
        self.assertFalse(are_strings_similar("1936", "2023"))

    # Casos mixtos
    def test_nombre_y_anio(self):
        self.assertTrue(are_strings_similar("María López 1936", "Maria Lopez, 1936"))

    def test_nombre_con_extras(self):
        self.assertTrue(are_strings_similar("José Antonio Primo de Rivera", "Jose Antonio Primo Rivera"))

    def test_organizacion_y_anio(self):
        self.assertTrue(are_strings_similar("CNT 1936", "C.N.T., 1936"))

    def test_espacios_y_mayusculas(self):
        self.assertTrue(are_strings_similar("  juan pérez ", "Juan Perez"))

    def test_cadenas_totalmente_distintas(self):
        self.assertFalse(are_strings_similar("UGT", "María López"))

    def test_una_vacia_otra_no(self):
        self.assertFalse(are_strings_similar("", "CNT"))

if __name__ == "__main__":
    unittest.main()

# No failures