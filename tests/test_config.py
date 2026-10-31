from app.config import Settings


def test_review_noise_control_defaults():
    settings = Settings(_env_file=None)

    assert settings.review_categories == "security,bug,performance,dependency,impact"
    assert settings.max_inline_comments == 15
