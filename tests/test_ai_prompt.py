from app.services.ai_prompt import (
    build_system_prompt,
    generate_boundary_tag,
    wrap_user_text,
)


def test_generate_boundary_tag_is_unpredictable_per_call():
    # Not cryptographically exhaustive - just confirms this isn't a
    # fixed/guessable string, which is the whole point of replacing the
    # old <text_to_rewrite> constant.
    assert generate_boundary_tag() != generate_boundary_tag()


def test_wrap_user_text_uses_the_given_tag():
    wrapped = wrap_user_text("Hello there", "abc123")

    assert wrapped == "<abc123>\nHello there\n</abc123>"


def test_build_system_prompt_embeds_instruction_and_tag():
    result = build_system_prompt("Correct the grammar.", "abc123")

    assert "Correct the grammar." in result
    assert "<abc123>" in result
    assert "</abc123>" in result


def test_build_system_prompt_with_different_tags_produces_different_boundaries():
    first = build_system_prompt("Correct the grammar.", "abc123")
    second = build_system_prompt("Correct the grammar.", "xyz789")

    assert "<abc123>" not in second
    assert "<xyz789>" not in first
