import unittest
from graph_utils import are_strings_similar
from ner_logic import ner_function 

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

# It not only does NER but also identifies possible node cases
class TestNER(unittest.TestCase):
    
    def assertNodesAlmostEqual(self, actual_list, expected_list):
        """
        Custom assertion to check if nodes are "almost equal".

        This test passes if for every item in `expected_list`, it can be found
        as a substring in at least one item of the `actual_list`.
        This makes the test robust to minor boundary or substring issues.
        """
        missing_nodes = []
        for expected_node in expected_list:
            is_found = False
            for actual_node in actual_list:
                if expected_node in actual_node:
                    is_found = True
                    break
            if not is_found:
                missing_nodes.append(expected_node)

        # If the list of missing nodes is not empty, the assertion fails.
        self.assertFalse(
            missing_nodes,
            f"The following expected nodes were not found in any form in the actual results {actual_list}:\n{missing_nodes}"
        )

    def test_nombre_persona(self):
        text = "María López fue elegida presidenta del comité."
        expected = ["María López", "presidenta", "comité"]
        self.assertNodesAlmostEqual(ner_function(text), expected)

    def test_lugar_simple(self):
        text = "La conferencia tuvo lugar en Madrid."
        expected = ["conferencia", "Madrid"]
        self.assertNodesAlmostEqual(ner_function(text), expected)

    def test_organizacion_abreviada(self):
        text = "El sindicato CNT organizó la huelga."
        expected = ["sindicato", "CNT", "huelga"]
        self.assertNodesAlmostEqual(ner_function(text), expected)

    def test_fecha_completa(self):
        text = "El 18 de julio de 1936 comenzó la Guerra Civil."
        expected = ["18 de julio de 1936", "1936", "Guerra Civil"]
        self.assertNodesAlmostEqual(ner_function(text), expected)

    def test_organizacion_con_puntos(self):
        text = "La C.N.T. fue una de las fuerzas principales."
        expected = ["C.N.T.", "fuerzas"]
        self.assertNodesAlmostEqual(ner_function(text), expected)

    def test_multiple_entidades(self):
        text = "Federico García Lorca nació en Granada en 1898."
        expected = ["Federico García Lorca", "Granada", "1898"]
        self.assertNodesAlmostEqual(ner_function(text), expected)

    def test_solo_fecha_año(self):
        text = "En 1936 estalló el conflicto."
        expected = ["1936", "conflicto"]
        self.assertNodesAlmostEqual(ner_function(text), expected)

    def test_organizacion_larga(self):
        text = "La Unión General de Trabajadores (UGT) convocó una manifestación."
        expected = ["Unión General de Trabajadores", "UGT", "manifestación"]
        self.assertNodesAlmostEqual(ner_function(text), expected)

    def test_persona_con_titulo(self):
        text = "El general Francisco Franco tomó el poder."
        expected = ["general", "Francisco Franco", "poder"]
        self.assertNodesAlmostEqual(ner_function(text), expected)

    def test_lugar_con_articulo(self):
        text = "Los combates comenzaron en El Escorial."
        expected = ["combates", "El Escorial"]
        self.assertNodesAlmostEqual(ner_function(text), expected)

    def test_caso_ambiguo(self):
        text = "Valencia apoyó la proclamación."
        expected = ["Valencia", "proclamación"]
        self.assertNodesAlmostEqual(ner_function(text), expected)

    def test_nombre_con_y(self):
        text = "Juan Ramón Jiménez y Marga Gil Roësset vivieron en Madrid."
        expected = ["Juan Ramón Jiménez", "Marga Gil Roësset", "Madrid"]
        self.assertNodesAlmostEqual(ner_function(text), expected)

    def test_siglas_y_fecha(self):
        text = "El PSOE fue fundado en 1879."
        expected = ["PSOE", "1879"]
        self.assertNodesAlmostEqual(ner_function(text), expected)
    
    def test_nombre_persona_largo(self):
        text = "Francisco Iacasta Catalán fue juzgado en Pamplona."
        expected = ["Francisco Iacasta Catalán", "Pamplona"]
        self.assertNodesAlmostEqual(ner_function(text), expected)

    def test_consejo_de_guerra(self):
        text = "Consejo de Guerra Ordinario de Plaza celebrado en Olite el 22 de enero de 1943."
        expected = ["Consejo de Guerra Ordinario de Plaza", "Olite", "22 de enero de 1943", "1943"]
        self.assertNodesAlmostEqual(ner_function(text), expected)

    def test_delito_auxilio_rebelion(self):
        text = "Fue acusado del delito de auxilio a la Rebelión."
        expected = ["delito", "auxilio a la Rebelión", "Rebelión"]
        self.assertNodesAlmostEqual(ner_function(text), expected)

    def test_ocupacion_y_edad(self):
        text = "El labriego de 46 años fue detenido."
        expected = ["labriego", "46 años"]
        self.assertNodesAlmostEqual(ner_function(text), expected)

    def test_ministerio_fiscal(self):
        text = "El Ministerio Fiscal solicitó pena de reclusión menor."
        expected = ["Ministerio Fiscal", "pena de reclusión menor", "reclusión menor"]
        self.assertNodesAlmostEqual(ner_function(text), expected)

    def test_pena_detallada(self):
        text = "Se le impuso la pena de reclusión menor de dos a nueve años."
        expected = ["pena de reclusión menor de dos a nueve años", "reclusión menor", "dos a nueve años"]
        self.assertNodesAlmostEqual(ner_function(text), expected)

    def test_sentencia_con_fecha(self):
        text = "El Consejo de Guerra dictó sentencia el 22 de enero de 1943 en Olite."
        expected = ["Consejo de Guerra", "sentencia", "22 de enero de 1943", "1943", "Olite"]
        self.assertNodesAlmostEqual(ner_function(text), expected)

    def test_nombre_autoridad_militar(self):
        text = "El tribunal estuvo presidido por el Teniente Coronel D. Rodrigo Torrent."
        expected = ["tribunal", "Teniente Coronel", "Rodrigo Torrent"]
        self.assertNodesAlmostEqual(ner_function(text), expected)

    def test_codigo_justicia_militar(self):
        text = "Fue condenado según el artículo 240 del Código de Justicia Militar de 1940."
        expected = ["artículo 240", "Código de Justicia Militar", "Código de Justicia Militar de 1940", "1940"]
        self.assertNodesAlmostEqual(ner_function(text), expected)

    def test_defensa_y_fallo(self):
        text = "La Defensa solicitó absolución o pena menor. Fallo del tribunal: dos años y seis meses de reclusión menor."
        expected = ["Defensa", "absolución", "pena menor", "Fallo", "tribunal", "dos años y seis meses de reclusión menor", "reclusión menor"]
        self.assertNodesAlmostEqual(ner_function(text), expected)

    def test_voto_particular(self):
        text = "Hubo un Voto particular proponiendo pena de un año de reclusión menor."
        expected = ["Voto particular", "pena de un año de reclusión menor", "reclusión menor", "un año"]
        self.assertNodesAlmostEqual(ner_function(text), expected)

    def test_vocales_del_consejo(self):
        text = "Los vocales del consejo votaron de forma unánime."
        expected = ["vocales", "consejo"]
        self.assertNodesAlmostEqual(ner_function(text), expected)

    def test_proceso_judicial_completo(self):
        text = "El acusado compareció ante el Juez Instructor el 15 de marzo."
        expected = ["acusado", "Juez Instructor", "15 de marzo"]
        self.assertNodesAlmostEqual(ner_function(text), expected)

    def test_cargos_y_condenas(self):
        text = "Fue condenado por sedición militar a 20 años de prisión."
        expected = ["sedición militar", "20 años", "prisión"]
        self.assertNodesAlmostEqual(ner_function(text), expected)

    def test_ubicaciones_y_eventos(self):
        text = "La batalla de Brunete tuvo lugar en julio de 1937."
        expected = ["batalla de Brunete", "Brunete", "julio de 1937", "1937"]
        self.assertNodesAlmostEqual(ner_function(text), expected)

    def test_instituciones_y_procedimientos(self):
        text = "El Tribunal Supremo ratificó la decisión del Consejo de Guerra."
        expected = ["Tribunal Supremo", "decisión", "Consejo de Guerra"]
        self.assertNodesAlmostEqual(ner_function(text), expected)

if __name__ == "__main__":
    unittest.main()

