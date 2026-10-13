"""Configuration loading tests for the AkadVerse orchestrator."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from app import config


class FakeModels:
  def __init__(self, names: list[str], supported_actions: dict[str, list[str]] | None = None):
    self._names = names
    self._supported_actions = supported_actions or {}

  def list(self):
    return [
      SimpleNamespace(
        name=f"models/{name}",
        supported_actions=self._supported_actions.get(name),
      )
      for name in self._names
    ]


class FakeClient:
  def __init__(self, names: list[str]):
    self.models = FakeModels(names)


class EnterpriseFakeClient:
  def __init__(self, names: list[str], supported_actions=None, **kwargs):
    self.models = FakeModels(names, supported_actions)
    self.kwargs = kwargs


def test_get_valid_model_name_prefers_configured_model(monkeypatch) -> None:
  monkeypatch.setattr(config.genai, "Client", lambda api_key: FakeClient(["gemini-2.0-flash", "gemini-2.0-flash-lite"]))

  result = config.get_valid_model_name("fake-key", "gemini-2.0-flash")

  assert result == "gemini-2.0-flash"


def test_get_valid_model_name_falls_back_when_preferred_is_missing(monkeypatch) -> None:
  monkeypatch.setattr(config.genai, "Client", lambda api_key: FakeClient(["gemini-2.0-flash-lite"]))

  result = config.get_valid_model_name("fake-key", "gemini-2.5-flash")

  assert result == "gemini-2.0-flash-lite"


def test_get_valid_model_name_uses_default_when_discovery_fails(monkeypatch) -> None:
  class BrokenModels:
    @staticmethod
    def list():
      raise RuntimeError("model listing failed")

  class BrokenClient:
    def __init__(self) -> None:
      self.models = BrokenModels()

  monkeypatch.setattr(config.genai, "Client", lambda api_key: BrokenClient())

  result = config.get_valid_model_name("fake-key", "gemini-2.5-flash")

  assert result == "gemini-2.5-flash"


def test_discover_enterprise_models_ranks_configured_models(monkeypatch) -> None:
  clients = []

  def build_client(**kwargs):
    client = EnterpriseFakeClient(
      ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.5-pro", "gemini-2.5-flash-image", "veo-3"],
      supported_actions={
        "gemini-2.5-flash": ["generateContent"],
        "gemini-2.5-flash-lite": ["generateContent"],
        "gemini-2.5-pro": ["generateContent"],
        "gemini-2.5-flash-image": ["generateImages"],
        "veo-3": ["generateVideos"],
      },
      **kwargs,
    )
    clients.append(client)
    return client

  monkeypatch.setattr(config.genai, "Client", build_client)

  result = config.discover_enterprise_model_names(
    project="test-project",
    location="us-central1",
    preferred_model="gemini-2.5-flash",
    fallback_models=["gemini-2.5-flash-lite"],
  )

  assert result[:2] == ["gemini-2.5-flash", "gemini-2.5-flash-lite"]
  assert "gemini-2.5-pro" not in result
  assert "gemini-2.5-flash-image" not in result
  assert "veo-3" not in result
  assert clients[0].kwargs == {
    "enterprise": True,
    "project": "test-project",
    "location": "us-central1",
  }


def test_load_settings_uses_enterprise_discovery(monkeypatch, tmp_path: Path) -> None:
  monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
  monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
  monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "europe-west4")
  monkeypatch.setattr(
    config,
    "discover_enterprise_model_names",
    lambda project, location, preferred_model, fallback_models: ["gemini-2.5-flash-lite", "gemini-2.5-flash"],
  )

  config_path = tmp_path / "config.yaml"
  config_path.write_text("llm:\n  provider: gemini_enterprise\n", encoding="utf-8")

  settings = config.load_settings(config_path)

  assert settings.llm_provider == "gemini_enterprise"
  assert settings.google_cloud_project == "test-project"
  assert settings.google_cloud_location == "europe-west4"
  assert settings.model_name == "gemini-2.5-flash-lite"
  assert settings.model_fallbacks == ["gemini-2.5-flash-lite", "gemini-2.5-flash"]


def test_load_settings_discovers_model_and_parses_tools(monkeypatch, tmp_path: Path) -> None:
  monkeypatch.setattr(config.genai, "Client", lambda api_key: FakeClient(["gemini-2.0-flash", "gemini-2.0-flash-lite"]))
  monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")

  config_path = tmp_path / "config.yaml"
  config_path.write_text(
    """
llm:
  provider: gemini
  model_name: gemini-2.5-flash
  model_fallbacks:
    - gemini-2.5-flash
    - gemini-2.0-flash
    - gemini-2.0-flash-lite
session:
  max_messages: 20
tools:
  - name: quiz_generator
    description: Generates quizzes
    parameters:
      type: object
      properties:
        topic:
          type: string
      required: [topic]
    endpoint: http://localhost:8016/quiz
    timeout_seconds: 12
    retry_count: 1
""".strip(),
    encoding="utf-8",
  )

  settings = config.load_settings(config_path)

  assert settings.llm_provider == "gemini"
  assert settings.model_name == "gemini-2.0-flash"
  assert settings.model_fallbacks == ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-2.0-flash-lite"]
  assert len(settings.tools) == 1
  assert settings.tools[0].name == "quiz_generator"
  assert settings.tools[0].endpoint == "http://localhost:8016/quiz"
  assert settings.tools[0].timeout_seconds == 12
  assert settings.tools[0].retry_count == 1
