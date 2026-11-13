import pytest

from app.services.factual_consistency_checker import check_factual_consistency


async def test_faithful_grammar_fix_is_consistent():
    result = await check_factual_consistency(
        "Thnks for your help", "Thank you for your help."
    )

    assert result is True


async def test_faithful_tone_change_is_consistent():
    result = await check_factual_consistency(
        "hey can u send me that file",
        "Could you please send me that file at your earliest convenience?",
    )

    assert result is True


async def test_faithful_question_rewrite_is_consistent():
    result = await check_factual_consistency(
        "why you are late for the meeting",
        "Why were you late for the meeting?",
    )

    assert result is True


async def test_changed_date_is_flagged_as_contradiction():
    result = await check_factual_consistency(
        "The meeting is on Tuesday at 3pm with John.",
        "The meeting is on Wednesday at 3pm with John.",
    )

    assert result is False


async def test_flipped_negation_is_flagged_as_contradiction():
    result = await check_factual_consistency(
        "I agree to the proposed terms.",
        "I do not agree to the proposed terms.",
    )

    assert result is False


# Broader validation: longer, multi-sentence paragraphs - a materially
# different regime from the single-sentence cases above, previously
# untested. All confirmed consistent with the existing threshold.


async def test_long_paragraph_grammar_fix_is_consistent():
    original = (
        "im writing to let you no that the shipment we ordered last "
        "week havent arrived yet and we was wondering if you could "
        "give us a update on when it will get here because our "
        "customer are asking us about it"
    )
    output = (
        "I'm writing to let you know that the shipment we ordered "
        "last week hasn't arrived yet, and we were wondering if you "
        "could give us an update on when it will get here, because "
        "our customers are asking us about it."
    )

    result = await check_factual_consistency(original, output)

    assert result is True


async def test_long_paragraph_formalization_is_consistent():
    original = (
        "Hey team, just a heads up that the server went down for "
        "about 20 minutes this morning around 9am because of a "
        "memory leak in the payment service. We fixed it by "
        "restarting the pods and we're monitoring it now. No "
        "customer data was lost."
    )
    output = (
        "This is to inform the team that the server experienced "
        "approximately 20 minutes of downtime this morning at "
        "approximately 9:00 AM, caused by a memory leak in the "
        "payment service. The issue was resolved by restarting the "
        "affected pods, and the system is currently being monitored. "
        "No customer data was affected."
    )

    result = await check_factual_consistency(original, output)

    assert result is True


async def test_sentence_reordering_is_consistent():
    original = (
        "The invoice is attached. Please pay within 30 days. "
        "The total amount due is $450."
    )
    output = (
        "The total amount due is $450. The invoice is attached, "
        "and payment is due within 30 days."
    )

    result = await check_factual_consistency(original, output)

    assert result is True


async def test_condensing_without_losing_facts_is_consistent():
    original = (
        "I wanted to reach out and let you know, first of all, that "
        "I really appreciate everything you've done, and also that "
        "the report you sent over is exactly what we needed, so "
        "thank you so much for that."
    )
    output = (
        "Thank you for the report you sent - it's exactly what we "
        "needed, and I really appreciate your effort."
    )

    result = await check_factual_consistency(original, output)

    assert result is True


# Broader validation: distortion types beyond date/negation - all
# newly confirmed as caught.


async def test_changed_amount_is_flagged_as_contradiction():
    result = await check_factual_consistency(
        "The total amount due is $450.",
        "The total amount due is $4500.",
    )

    assert result is False


async def test_changed_name_is_flagged_as_contradiction():
    result = await check_factual_consistency(
        "Please contact John Smith regarding the contract.",
        "Please contact Mark Smith regarding the contract.",
    )

    assert result is False


async def test_changed_location_is_flagged_as_contradiction():
    result = await check_factual_consistency(
        "The conference will be held in Chicago this year.",
        "The conference will be held in Denver this year.",
    )

    assert result is False


async def test_reversed_decision_is_flagged_as_contradiction():
    result = await check_factual_consistency(
        "Your application has been approved.",
        "Your application has been rejected.",
    )

    assert result is False


async def test_quantity_change_is_flagged_as_contradiction():
    result = await check_factual_consistency(
        "We shipped 120 units to the warehouse on Monday.",
        "We shipped 12 units to the warehouse on Monday.",
    )

    assert result is False


async def test_added_detail_is_not_flagged_as_contradiction():
    # A pure addition (not present in the original, but not
    # contradicting it either) is a known, deliberate blind spot: this
    # check only catches contradictions, not unsupported additions -
    # see check_factual_consistency's docstring.
    result = await check_factual_consistency(
        "The meeting is at 3pm.",
        "The meeting is at 3pm in the main conference room.",
    )

    assert result is True


@pytest.mark.xfail(
    reason=(
        "Confirmed model limitation: hedged/modal phrasing (e.g. "
        "'may have been') collapses the contradiction score even when "
        "the underlying claim is flipped. Isolated diagnosis: 'No "
        "customer data was affected.' -> 'Customer data was affected.' "
        "scores 0.996 (correctly caught); the same flip through 'Some "
        "customer data may have been affected.' scores 0.009 (missed) "
        "- the hedge, not sentence length or quantifier wording, is "
        "the cause. strict=True so this fails loudly (XPASS) if a "
        "future model swap or threshold change fixes it, rather than "
        "silently going stale."
    ),
    strict=True,
)
async def test_hedged_distortion_is_a_known_gap():
    result = await check_factual_consistency(
        "No customer data was affected.",
        "Some customer data may have been affected.",
    )

    assert result is False


async def test_fails_open_when_classifier_unavailable(monkeypatch):
    import app.services.factual_consistency_checker as module

    monkeypatch.setattr(module, "_classifier", None)

    result = await check_factual_consistency("original", "output")

    assert result is None
