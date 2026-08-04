"""Unit-тесты guardrails — без LLM/Docker, гоняются за секунды."""

from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from app.core.guardrails import (
    RetrievalGate,
    AnswerFaithfulnessGuard,
    AtomicClaim,
    ClaimEntailment,
    EntailmentCheck,
    FaithfulnessVerdict,
    doc_relevance_score,
    is_spelling_variant_in_context,
    is_quote_grounded,
)


class _FakeStructuredLLM:
    """Подменяет AnswerFaithfulnessGuard.llm — .with_structured_output(schema) возвращает
    заранее заданный ответ для этой схемы, без сети. Guard делает ДВА разных структурированных
    вызова (FaithfulnessVerdict, потом EntailmentCheck) на одном self.llm — фейк различает их
    по запрошенной схеме, а не просто отдаёт один и тот же ответ на любой вызов."""

    def __init__(self, responses: dict):
        self._responses = responses
        self.calls: list = []

    def with_structured_output(self, schema):
        response = self._responses[schema]

        async def _fn(prompt_value):
            self.calls.append((schema, prompt_value))
            return response

        return RunnableLambda(_fn)


def _doc(score: float, *, rerank: bool = True) -> Document:
    key = "rerank_score" if rerank else "hybrid_score"
    return Document(page_content="chunk", metadata={key: score})


def test_doc_relevance_prefers_rerank():
    d = Document(page_content="x", metadata={"rerank_score": 0.9, "hybrid_score": 0.1})
    assert doc_relevance_score(d) == 0.9


def test_gate_rejects_offtopic_scores():
    """Симулируем кейс Губки Боба: rerank ~0.50 при пороге 0.55."""
    gate = RetrievalGate(min_score=0.55)
    docs = [_doc(0.5011), _doc(0.5006), _doc(0.5004)]
    kept, meta = gate.filter_docs(docs)
    assert kept == []
    assert meta["passed"] is False
    assert meta["reason"] == "below_similarity_threshold"


def test_gate_keeps_confident_docs():
    gate = RetrievalGate(min_score=0.55)
    docs = [_doc(0.82), _doc(0.40), _doc(0.91)]
    kept, meta = gate.filter_docs(docs)
    assert meta["passed"] is True
    assert len(kept) == 2  # 0.40 отфильтрован
    assert meta["max_score"] == 0.91


def test_gate_empty_retrieval():
    gate = RetrievalGate(min_score=0.55)
    kept, meta = gate.filter_docs([])
    assert kept == []
    assert meta["reason"] == "empty_retrieval"


def test_gate_uses_looser_threshold_without_rerank():
    """Без reranker'а в metadata только hybrid_score (ранговый RRF-скор,
    не семантический) — тот же порог 0.55, что и для rerank_score, отсекал
    бы даже нормальные по теме вопросы (см. регресс, найденный при живом
    прогоне: 'Кто такие Адептус Астартес?' и оффтоп давали одинаковый
    max_score=0.5). Порог для hybrid-only должен быть отдельным и мягче."""
    gate = RetrievalGate(min_score=0.55, min_score_no_rerank=0.2)
    docs = [_doc(0.5, rerank=False)]
    kept, meta = gate.filter_docs(docs)
    assert meta["passed"] is True
    assert meta["scoring"] == "hybrid_only"
    assert meta["min_score"] == 0.2
    assert kept == docs


def test_gate_still_rejects_near_empty_hybrid_signal():
    gate = RetrievalGate(min_score=0.55, min_score_no_rerank=0.2)
    docs = [_doc(0.05, rerank=False)]
    kept, meta = gate.filter_docs(docs)
    assert kept == []
    assert meta["passed"] is False
    assert meta["scoring"] == "hybrid_only"


def test_gate_mixed_docs_use_rerank_threshold_when_any_reranked():
    """Если хотя бы часть документов прошла реранк — считаем скор reranked-шкалой."""
    gate = RetrievalGate(min_score=0.55, min_score_no_rerank=0.2)
    docs = [_doc(0.6, rerank=True), _doc(0.5, rerank=False)]
    kept, meta = gate.filter_docs(docs)
    assert meta["scoring"] == "rerank"
    assert meta["min_score"] == 0.55


def test_faithfulness_skipped_on_high_confidence():
    guard = AnswerFaithfulnessGuard(enabled=True)
    docs = [_doc(0.85)]
    need, meta = guard.should_verify(docs, degraded=False)
    assert need is False
    assert meta["reason"] == "high_retrieval_confidence"


def test_faithfulness_required_in_gray_zone():
    guard = AnswerFaithfulnessGuard(enabled=True)
    docs = [_doc(0.60)]
    need, meta = guard.should_verify(docs, degraded=False)
    assert need is True
    assert meta["skipped"] is False


def test_faithfulness_required_when_degraded():
    guard = AnswerFaithfulnessGuard(enabled=True)
    docs = [_doc(0.95)]
    need, meta = guard.should_verify(docs, degraded=True)
    assert need is True
    assert meta["reason"] == "degraded_pipeline"


def test_spelling_variant_detected():
    """Регресс: верификатор помечал 'Ярик' как unsupported_claim,
    хотя контекст описывает 'Себастьян Яррик' — та же сущность, опечатка."""
    context = "Себастьян Яррик — легендарный лорд-комиссар Стального Легиона."
    assert is_spelling_variant_in_context("Ярик", context) is True


def test_spelling_variant_rejects_unrelated_claim():
    context = "Себастьян Яррик — легендарный лорд-комиссар Стального Легиона."
    assert is_spelling_variant_in_context("Магнус Красный", context) is False


def test_quote_grounded_on_short_context():
    context = "Себастьян Яррик — легендарный лорд-комиссар Стального Легиона."
    quote = "Себастьян Яррик — легендарный лорд-комиссар Стального Легиона."
    assert is_quote_grounded(quote, context) is True


# Реальный ContextBuilder-вывод (5 чанков, дословно с живого Qdrant) на вопрос
# "Кто такая Вилиар Ифисс?" — намеренно не упрощённый синтетический пример: баг ниже
# не воспроизводится на коротких/менее "накладывающихся" текстах (несколько чанков
# пересказывают одни и те же фразы про Вилиар с небольшими вариациями, и именно это
# сочетание длины и повторов сбивает эвристику autojunk — на выдуманном тексте с
# похожей длиной, но без такого пересечения, coverage остаётся 1.0 даже без фикса).
_REAL_LONG_CONTEXT = (
    "СТАТЬЯ: Вилиар Ифисс\nКОНТЕКСТ: Вилиар Ифисс\n\n"
    "Суккуб Вилиар Ифисс — большая знаменитость и одна из лучших гладиатрис Комморры. "
    "Она вооружена агонизатором и бласт-пистолетом, что помогает ей расправляться как с "
    "чудовищными тварями и тяжёлой техникой, так и с рядовыми солдатами любой армии.\n\n"
    "СТАТЬЯ: Вилиар Ифисс\nКОНТЕКСТ: Вилиар Ифисс > История\n\n"
    "С давних пор больше всего комморритов собирается, чтобы посмотреть кульминационную "
    "схватку, в которой принимает участие не кто иная, как Королева Ножей собственной "
    "персоной, она же Лелит Гесперакс. За всю историю цивилизации тёмных эльдар не было "
    "гладиатора, обладающего более впечатляющими способностями. Таковы проворность и "
    "грациозность Лелит, что для защиты ей не нужны доспехи — направленные в неё удары "
    "просто проходят в воздухе. Что удивительно, в культе Распри есть и другое дарование. "
    "На самом деле, если бы не, кажется, безграничные таланты Лелит Гесперакс, то её "
    "соратница-суккуб, Вилиар Ифисс, вполне могла бы стать самой смертоносной ведьмой. "
    "Несмотря на завоевание места в правящей триаде суккубов культа Распри, Вилиар всегда "
    "находилась в тени лучшей из них. И хотя пока что им не доводилось скрещивать клинки, "
    "в игорных домах Комморры можно сделать ставку на потенциального победителя, если эти "
    "двое когда-нибудь всё же найдут подходящий повод для драки\n\n"
    "СТАТЬЯ: Культ Распри\nКОНТЕКСТ: Культ Распри > Список армии культа Распри > Командующие\n\n"
    "### Командующие  \n"
    "* **Лелит Гесперакс** — прима-суккуба, фактически глава культа.\n"
    "* Вилиар Ифисс «Принцесса Боли» — суккуба Вилиар, большая знаменитость и одна из "
    "лучших гладиатрис Комморры. Она вооружена агонизатором и бласт-пистолетом, что "
    "помогает ей расправляться как с чудовищными тварями и тяжёлой техникой, так и с "
    "рядовыми солдатами любой армии. Командующая отряда ведьм известных как Дочери Бойни.\n"
    "* **Элира Нарцистин** — не так давно ставшая суккубом в культе Распри, Элира заключила "
    "соглашение с архонтом Вреском в надежде выйти из тени леди Гесперакс.\n"
    "* **Мэлинн** — командующая небольшого отряда ведьм известного как Поработители.\n"
    "* **Шайликс** — лидер отряда Последовательниц Кхейна.\n\n"
    "СТАТЬЯ: Вилиар Ифисс\nКОНТЕКСТ: Вилиар Ифисс > История > Славу нужно заслужить\n\n"
    "### Славу нужно заслужить  \n"
    "Из всей руководящей триады суккубов чаще всего именно Вилиар представляет интересы "
    "культа во время рейдов в реальное пространство. И хотя тщеславные архонты сильнее "
    "всего хотят видеть Лелит Гесперакс в проводимых ими налётах, Королева Ножей редко "
    "покидает арены Комморры. Вилиар же, наоборот, обычно не упускает возможности "
    "продемонстрировать своё мастерство. К тому же ей хорошо известно, что на полях "
    "сражений материального мира можно снискать великую славу.  \n"
    "Подвиги Вилиар в ходе таких вторжений заработали ей немалую известность, и альянсы, "
    "что она организовала, наделили её огромным влиянием в Тёмном городе. Злопыхатели "
    "утверждают, будто она пытается заручиться достаточной поддержкой, чтобы свергнуть "
    "свою главную соперницу. Если Лелит и знает об этом, то не подаёт вида. Многие "
    "заявляют, что она просто не видит в этом нужды.\n\n"
    "СТАТЬЯ: Вилиар Ифисс\nКОНТЕКСТ: Вилиар Ифисс > История > Звёздный час\n\n"
    "Только сейчас зеленокожие осознали, что к драке присоединился другой, более опасный "
    "соперник. Этого короткого замешательства хватило оставшимся ведьмам, чтобы обратить "
    "ход сражения в свою пользу. Они ринулись на чужаков с новыми силами и теперь были "
    "практически неуязвимы, благодаря тому, что уже насытились болью и смертью. Вилиар "
    "видела, как Эльдира, её ближайшая последовательница, умело перерезает глотки и "
    "отрубает конечности, демонстрируя высочайшие навыки обращения с силовым клинком. "
    "Неподалёку, её кровная сестра Эльвира с точностью хирурга сдирала у орков мясо с "
    "костей при помощи бритвоцепов. Из-за столь неожиданного всплеска насилия орки "
    "запаниковали, чем только сильнее раззадорило гекатарий, достигших новых высот "
    "исступлённой ярости.\n"
    "Несмотря на обстоятельства, побудившие Вилиар вмешаться, эта схватка начинала ей "
    "по-настоящему нравиться._»  \n"
    "– Суккуб Вилиар Ифисс сражается вместе со своими ученицами против орков на арене "
    "«Крусибаэль»"
)


def test_quote_grounded_verbatim_quote_in_real_long_context():
    """Регресс: difflib.SequenceMatcher по умолчанию (autojunk=True) резал
    find_longest_match до ~70% на этом реальном ContextBuilder-выводе даже для цитаты,
    дословно и полностью содержащейся в контексте (эвристика "популярные символы —
    мусор" исключает их из совпадающих блоков) — AnswerFaithfulnessGuard ложно
    отказывал в полностью подтверждённом ответе на живой вопрос 'Кто такая Вилиар
    Ифисс?'. Без autojunk=False в is_quote_grounded этот тест падает."""
    quote = (
        "Вилиар Ифисс «Принцесса Боли» — суккуба Вилиар, большая знаменитость и одна из "
        "лучших гладиатрис Комморры. Она вооружена агонизатором и бласт-пистолетом, что "
        "помогает ей расправляться как с чудовищными тварями и тяжёлой техникой, так и с "
        "рядовыми солдатами любой армии. Командующая отряда ведьм известных как Дочери Бойни."
    )
    assert len(_REAL_LONG_CONTEXT) >= 200
    assert is_quote_grounded(quote, _REAL_LONG_CONTEXT) is True


def test_quote_grounded_rejects_fabricated_quote_in_real_long_context():
    fabricated = "Вилиар Ифисс — примарх Тысячи Сынов и мастер тайных искусств Тизариума."
    assert is_quote_grounded(fabricated, _REAL_LONG_CONTEXT) is False


def test_structured_output_schemas_have_class_docstrings():
    """Регресс: GigaChat отдаёт 'Incorrect function or tool description. Description is
    required' на with_structured_output(), если у pydantic-схемы нет docstring'а (он
    становится description в JSON-схеме тула) — поймано на первом же живом вызове с
    ClaimEntailment/EntailmentCheck, у которых были только Field(description=...) на
    полях, но не на классе."""
    for schema in (FaithfulnessVerdict, AtomicClaim, ClaimEntailment, EntailmentCheck):
        assert schema.__doc__, f"{schema.__name__} needs a class docstring (GigaChat schema description)"


# --------------------------------------------------------------------------- _check_entailment


async def test_check_entailment_empty_list_skips_llm_call():
    guard = AnswerFaithfulnessGuard(enabled=True)
    guard.llm = _FakeStructuredLLM({})  # если бы дёрнули LLM, KeyError на пустом dict
    assert await guard._check_entailment([]) == []


async def test_check_entailment_returns_verdicts_in_order():
    guard = AnswerFaithfulnessGuard(enabled=True)
    fake_result = EntailmentCheck(verdicts=[ClaimEntailment(entailed=True), ClaimEntailment(entailed=False)])
    guard.llm = _FakeStructuredLLM({EntailmentCheck: fake_result})
    claims = [
        AtomicClaim(claim="a", supporting_quote="qa"),
        AtomicClaim(claim="b", supporting_quote="qb"),
    ]
    assert await guard._check_entailment(claims) == [True, False]


async def test_check_entailment_fails_safe_on_verdict_count_mismatch():
    guard = AnswerFaithfulnessGuard(enabled=True)
    # Модель вернула вердиктов меньше, чем claims — не доверяем частичному ответу.
    fake_result = EntailmentCheck(verdicts=[ClaimEntailment(entailed=True)])
    guard.llm = _FakeStructuredLLM({EntailmentCheck: fake_result})
    claims = [
        AtomicClaim(claim="a", supporting_quote="qa"),
        AtomicClaim(claim="b", supporting_quote="qb"),
    ]
    assert await guard._check_entailment(claims) == [False, False]


async def test_check_entailment_fails_safe_on_llm_error():
    class _BrokenLLM:
        def with_structured_output(self, schema):
            raise RuntimeError("boom")

    guard = AnswerFaithfulnessGuard(enabled=True)
    guard.llm = _BrokenLLM()
    claims = [AtomicClaim(claim="a", supporting_quote="qa")]
    assert await guard._check_entailment(claims) == [False]


# --------------------------------------------------------------------------- verify() integration


async def test_verify_rejects_real_but_mismatched_quote():
    """Регресс-сценарий, ради которого добавлен второй LLM-вызов: цитата дословно
    существует в контексте (слой 1 проходит), но подтверждает другой номер легиона,
    чем заявлено в claim'е (слой 2 — entailment — должен это поймать)."""
    guard = AnswerFaithfulnessGuard(enabled=True, min_score=1.0)
    doc = Document(
        page_content="Магнус — примарх XV легиона Тысячи Сынов.",
        metadata={"rerank_score": 0.9},
    )
    verdict = FaithfulnessVerdict(
        claims=[
            AtomicClaim(
                claim="Магнус — примарх XVII легиона",
                supporting_quote="Магнус — примарх XV легиона Тысячи Сынов.",
            )
        ],
        reasoning="цитата найдена в контексте",
    )
    guard.llm = _FakeStructuredLLM(
        {
            FaithfulnessVerdict: verdict,
            EntailmentCheck: EntailmentCheck(verdicts=[ClaimEntailment(entailed=False)]),
        }
    )

    passed, _verdict, meta = await guard.verify("вопрос", "ответ", [doc])

    assert passed is False
    assert meta["unsupported_claims"] == ["Магнус — примарх XVII легиона"]


async def test_verify_passes_when_quote_grounded_and_entailed():
    guard = AnswerFaithfulnessGuard(enabled=True, min_score=1.0)
    doc = Document(
        page_content="Магнус — примарх XV легиона Тысячи Сынов.",
        metadata={"rerank_score": 0.9},
    )
    verdict = FaithfulnessVerdict(
        claims=[
            AtomicClaim(
                claim="Магнус — примарх XV легиона",
                supporting_quote="Магнус — примарх XV легиона Тысячи Сынов.",
            )
        ],
        reasoning="всё сходится",
    )
    guard.llm = _FakeStructuredLLM(
        {
            FaithfulnessVerdict: verdict,
            EntailmentCheck: EntailmentCheck(verdicts=[ClaimEntailment(entailed=True)]),
        }
    )

    passed, _verdict, meta = await guard.verify("вопрос", "ответ", [doc])

    assert passed is True
    assert meta["unsupported_claims"] == []


async def test_verify_skips_entailment_call_for_claims_with_fabricated_quote():
    """Оптимизация: если цитата уже не прошла слой 1 (её нет в контексте), entailment
    на неё не тратится — EntailmentCheck-ответ подгружен только для claim'ов, реально
    дошедших до слоя 2 (здесь их 0)."""
    guard = AnswerFaithfulnessGuard(enabled=True, min_score=1.0)
    doc = Document(page_content="Совсем другой текст ни о чём.", metadata={"rerank_score": 0.9})
    verdict = FaithfulnessVerdict(
        claims=[AtomicClaim(claim="Выдуманный факт", supporting_quote="Полностью придуманная цитата")],
        reasoning="...",
    )
    fake_llm = _FakeStructuredLLM(
        {
            FaithfulnessVerdict: verdict,
            EntailmentCheck: EntailmentCheck(verdicts=[]),
        }
    )
    guard.llm = fake_llm

    passed, _verdict, meta = await guard.verify("вопрос", "ответ", [doc])

    assert passed is False
    assert meta["unsupported_claims"] == ["Выдуманный факт"]
    assert all(schema is not EntailmentCheck for schema, _ in fake_llm.calls)
