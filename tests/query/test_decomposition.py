from vacxis_capstone_fa26.query import QueryProcessingConfig, QueryProcessor


def processor(**kwargs):
    return QueryProcessor(QueryProcessingConfig(**kwargs), vaccine_mapping={"a": "Verorab"})


def test_complex_query_preserves_all_medical_context():
    query = "Trẻ 2 tháng tuổi không bị dị ứng trứng có tiêm được Verorab không và có những phản ứng phụ nào?"
    result = processor().process(query)
    assert len(result.subqueries) == 2
    assert result.search_queries == result.subqueries
    assert all(query in q for q in result.subqueries)
    assert result.fallback_query == query


def test_simple_query_not_decomposed():
    assert processor().process("Verorab có tác dụng phụ nào?").subqueries == ()


def test_equivalent_focus_clauses_deduplicated():
    result = processor().process("Verorab có tiêm được không và có tác dụng phụ nào và có phản ứng phụ nào?")
    assert len(result.subqueries) == 2
    assert len(set(result.subqueries)) == 2


def test_limit_never_drops_remaining_question():
    query = "Verorab có tiêm được không và có phản ứng phụ nào và lịch tiêm thế nào?"
    result = processor(max_subqueries=2).process(query)
    assert result.subqueries == ()
    assert result.search_queries == (query,)
    assert any(d.code == "subquery_limit" for d in result.diagnostics)


def test_multiple_subjects_are_not_mixed():
    query = "Bé 2 tháng tuổi không bị dị ứng và mẹ 30 tuổi bị dị ứng có tiêm được Verorab không và có phản ứng phụ nào?"
    result = processor().process(query)
    assert result.subqueries == ()
    assert result.understanding.multiple_subjects
    assert result.search_queries == (query,)


def test_conjunction_in_conditions_not_split():
    result = processor().process("Tôi có bệnh nền và dị ứng trứng, Verorab chống chỉ định khi nào?")
    assert result.subqueries == ()


def test_feature_disabled():
    result = processor(enable_decomposition=False).process("Verorab có tiêm được không và có phản ứng phụ nào?")
    assert result.subqueries == ()
