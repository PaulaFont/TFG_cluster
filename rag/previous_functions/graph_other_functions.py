# TODO: Apply NER in here. (Future idea: to provide information to the node as, the type: PER, LOC, ORG, DATE, ...)
def manage_wrongs(list_text):
    new_triplets = []
    for text in list_text:
        entities = [ent.text for ent in nlp(text).ents if ent is not None]
        entity_positions = []
        for e in entities:
            match = re.search(re.escape(e), text)
            if match:
                entity_positions.append((e, match.start()))
            else:
                print(f"Warning: entity '{e}' not found in text.")

        entity_positions.sort(key=lambda x: x[1])

        triplets = []
        current_subject = None

        for i in range(len(entity_positions) - 1):
            s_entity, s_idx = entity_positions[i]
            o_entity, o_idx = entity_positions[i + 1]

            # If no subject yet, assume first entity is the subject
            if current_subject is None:
                current_subject = s_entity

            # Extract relation
            relation = text[s_idx + len(s_entity):o_idx].strip(" ,.")

            # Check if the relation contains a verb or makes grammatical sense
            if any(verb in relation for verb in ['was', 'served', 'became', 'is', 'are']):
                triplets.append((current_subject, relation, o_entity))

                # Update subject if it's a new clause with a new actor
                if 'and' in relation or ',' in relation:
                    current_subject = current_subject  # optional: update if needed
        new_triplets.append(triplets)


# Idea: to manage long edges somehow (AT THE MOMENT WE WILL SKIP THEM)    
def create_intermediate_triplets(subject: str, long_predicate: str, obj: str):# -> List[Tuple[str, str, str]]:
    """Create intermediate nodes/triplets for long predicates"""
    
    # Try to extract entities from the long predicate
    entities = [ent.text for ent in nlp(long_predicate).ents if ent is not None]

    
    if entities:
        # Use entities as intermediate nodes
        triplets = []
        current_subj = subject
        
        for i, entity in enumerate(entities):
            if i == len(entities) - 1:
                # Last entity connects to final object
                relation = "leads_to"
                triplets.append((current_subj, relation, obj))
            else:
                # Connect through intermediate entity
                relation = "connected_via"
                triplets.append((current_subj, relation, entity))
                current_subj = entity
        
        return triplets
    
    # Try to split by conjunctions and create chain
    parts = re.split(r'\s+and\s+|\s*,\s+|\s+then\s+', long_predicate)
    parts = [p.strip() for p in parts if p.strip()]
    
    if len(parts) > 1:
        triplets = []
        current_subj = subject
        
        for i, part in enumerate(parts):
            short_part = shorten_with_ner(part, MAX_LEN_EDGE)
            
            if i == len(parts) - 1:
                # Last part connects to final object
                triplets.append((current_subj, short_part, obj))
            else:
                # Create intermediate node
                intermediate_node = f"intermediate_{hash(part) % 1000}"
                triplets.append((current_subj, short_part, intermediate_node))
                current_subj = intermediate_node
        
        return triplets
    
    # Fallback: just shorten the predicate
    short_predicate = shorten_with_ner(long_predicate, MAX_LEN_EDGE)
    return [(subject, short_predicate, obj)]



def filter_and_fix_triplets_old(current_graph, triplets):
    #TODO: remove, now it's temporary
    existing_triplets = [
        # About the Person (Subject)
        ('Francisco Iacasta Catalán', 'labriego', 'era'),
        ('Francisco Iacasta Catalán', '46 años', 'tenía'),
        ('Francisco Iacasta Catalán', 'Olite', 'era_natural_de'),
        ('Francisco Iacasta Catalán', 'Olite', 'era_vecino_de'),

        # The Trial and Accusation
        ('Francisco Iacasta Catalán', 'Consejo de Guerra', 'fue_juzgado_por'),
        ('Consejo de Guerra', 'Pamplona', 'tuvo_lugar_en'),
        ('Consejo de Guerra', '22 de enero de 1943', 'fecha_de_juicio'),
        ('Francisco Iacasta Catalán', 'auxilio a la Rebelión', 'fue_acusado_de'),

        # The Court's Composition
        ('Consejo de Guerra', 'Rodrigo Torrent', 'fue_presidido_por'),
        ('Rodrigo Torrent', 'Teniente Coronel', 'tenía_título_de'),
        ('Fallo', 'vocales del consejo', 'fue_confirmado_por'),

        # Requests from Prosecution and Defense
        ('Ministerio Fiscal', 'reclusión menor de dos a nueve años', 'solicitó_pena'),
        ('Defensa', 'absolución o pena menor', 'solicitó'),

        # The Verdict and Sentence
        ('Consejo de Guerra', 'Francisco Iacasta Catalán', 'condenó'),
        ('Consejo de Guerra', 'dos años y seis meses de reclusión menor', 'impuso_pena'),
        ('Condena', 'artículo 240 del Código de Justicia Militar', 'basada_en'),

        # Dissenting Opinion
        ('Voto particular', 'un año de reclusión menor', 'propuso_pena')
    ]
    existing_nodes = list(set([t[0] for t in existing_triplets] + [t[1] for t in existing_triplets]))

    triplets = current_graph.edges(keys=True) #node, node, edge #TODO: fix later
    print(f"Orignal triplets to add: {triplets}")
    current_triplets = []
    for s, o, e in triplets:  # TODO: check order that triplets come from the LLM
        if filter_edge(e):  # Short edge
            sub_match = find_similars(existing_nodes, s)
            ob_match = find_similars(existing_nodes, o)

            subj_final = sub_match if sub_match else s
            obj_final = ob_match if ob_match else o

            if sub_match:
                print(f"Matched subject '{s}' to existing node '{sub_match}'")
            if ob_match:
                print(f"Matched object '{o}' to existing node '{ob_match}'")

            current_triplets.append((subj_final, e, obj_final))
            
            temporary_new_triplets = []
            # NER (new triplets)
            if not sub_match:
                sub_list = ner_function(s)
                if sub_list and sub_list != [s]:
                    print(f"NER for subject '{s}': {sub_list}")
                    new_triplets_sub = link_components_by_context(s, sub_list)
                    temporary_new_triplets += new_triplets_sub
            if not ob_match:
                ob_list = ner_function(o)
                if ob_list and ob_list != [o]:
                    print(f"NER for object '{o}': {ob_list}")
                    new_triplets_ob = link_components_by_context(o, ob_list)         
                    temporary_new_triplets += new_triplets_ob
            
            # Check if the new nodes already exist:
            for s, o, e in temporary_new_triplets: 
                sub_match = find_similars(existing_nodes, s)
                ob_match = find_similars(existing_nodes, o)

                subj_final = sub_match if sub_match else s
                obj_final = ob_match if ob_match else o

                if sub_match:
                    print(f"Matched subject '{s}' to existing node '{sub_match}'")
                if ob_match:
                    print(f"Matched object '{o}' to existing node '{ob_match}'")

                current_triplets.append((subj_final, e, obj_final))

        else: 
            print(f"Removed triplet for LONG EDGE ({s} {e} {o})")
    print(f"New triplets to add: {current_triplets}")
    return current_triplets

