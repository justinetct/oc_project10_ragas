"""Tests du mode LLM→SQL et de ses garde-fous de génération SQL (utils/sql/llm_sql_*).

Aucun appel API Mistral : la génération du LLM est SIMULÉE (monkeypatch de
`generate_sql`). On vérifie :
- la validation du schéma de sortie Pydantic (`LlmSqlDecision`) ;
- le refus statique d'une requête dangereuse (générée ou simulée) ;
- la conservation du mode contrôlé par défaut (`SQL_GENERATION_MODE=controlled`) ;
- la compatibilité du routeur (mode contrôlé inchangé, mode llm passe par le SQL Tool) ;
- l'absence de casse de `scripts/evaluate_ragas.py`.

Les exécutions SQL se font sur une base SQLite TEMPORAIRE (jamais la base du projet).
"""

import inspect
import os
import py_compile
import sqlite3

# Une clé factice suffit : ces tests n'appellent jamais l'API (load_dotenv n'override pas).
os.environ.setdefault("MISTRAL_API_KEY", "test-key-not-used")

import pytest
from pydantic import ValidationError

from utils import config
import utils.router as router
import utils.sql.sql_tool as sql_tool
import utils.sql.llm_sql_pipeline as pipeline
from utils.sql.llm_sql_generator import (
    LlmSqlDecision,
    validate_read_only_sql,
)


# --- 1. Schéma de sortie Pydantic (LlmSqlDecision) ----------------------------

def test_decision_accepts_valid_query():
    """Une décision valide (should_query=True + requête SELECT) est acceptée telle quelle."""
    decision = LlmSqlDecision(
        should_query=True,
        sql="SELECT player_name FROM players LIMIT 5",
        reason="classement simple",
        expected_result_type="classement",
    )
    assert decision.should_query is True
    assert decision.sql == "SELECT player_name FROM players LIMIT 5"


def test_decision_strips_sql_whitespace():
    """Les espaces autour de la requête générée sont retirés (nettoyage avant exécution)."""
    decision = LlmSqlDecision(should_query=True, sql="  SELECT 1  ", reason="x")
    assert decision.sql == "SELECT 1"


def test_decision_true_without_sql_is_rejected():
    """should_query=True sans requête est incohérent : la validation Pydantic échoue."""
    with pytest.raises(ValidationError):
        LlmSqlDecision(should_query=True, sql=None, reason="incohérent")
    with pytest.raises(ValidationError):
        LlmSqlDecision(should_query=True, sql="   ", reason="vide")


def test_decision_false_nullifies_sql():
    """Un refus (should_query=False) ne porte jamais de requête, même si le LLM en propose une."""
    decision = LlmSqlDecision(should_query=False, sql="SELECT 1", reason="hors périmètre")
    assert decision.sql is None


def test_decision_requires_reason():
    """Une décision sans justification (`reason` vide) est refusée : on exige toujours une raison."""
    with pytest.raises(ValidationError):
        LlmSqlDecision(should_query=False, sql=None, reason="")


# --- 2. Refus statique d'une requête dangereuse -------------------------------

@pytest.mark.parametrize(
    "dangerous_sql",
    [
        "DROP TABLE players",
        "DELETE FROM players",
        "UPDATE stats SET points = 0",
        "INSERT INTO players (player_name) VALUES ('x')",
        "PRAGMA table_info(players)",
        "SELECT 1; DROP TABLE players",
        "WITH x AS (SELECT 1) INSERT INTO players SELECT 'hack'",
        "ALTER TABLE players ADD COLUMN hacked TEXT",
    ],
)
def test_validate_refuses_dangerous_sql(dangerous_sql):
    """Toute requête générée non lecture-seule est refusée par le contrôle statique."""
    with pytest.raises(ValueError):
        validate_read_only_sql(dangerous_sql)


def test_validate_accepts_read_only_sql():
    """Une requête en lecture seule (SELECT, ou WITH ... SELECT) passe la validation et est renvoyée telle quelle."""
    assert validate_read_only_sql("SELECT player_name FROM players LIMIT 5") == \
        "SELECT player_name FROM players LIMIT 5"
    assert validate_read_only_sql(
        "WITH t AS (SELECT points FROM stats) SELECT COUNT(*) FROM t"
    ).startswith("WITH")


# --- 3. Mode par défaut : LLM→SQL (approche recommandée) ----------------------

def test_default_sql_generation_mode_is_llm():
    """Le défaut code de la génération SQL est `llm` (approche agent + Tool retenue).

    On lit `config` (résolu à l'import), que la fixture autouse de mode contrôlé ne
    touche pas — le test reflète donc bien le défaut code, pas le réglage des tests.
    Le mode `controlled` reste un benchmark sécurisé, activable par configuration.
    """
    assert config.SQL_GENERATION_MODE == "llm"


# --- Base NBA temporaire (pour les exécutions du pipeline / routeur) -----------

@pytest.fixture
def nba_db(tmp_path):
    path = str(tmp_path / "nba_llm.sqlite")
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE teams (team_code TEXT PRIMARY KEY, team_name TEXT NOT NULL);
        CREATE TABLE players (
            player_id INTEGER PRIMARY KEY,
            player_name TEXT NOT NULL UNIQUE,
            team_code TEXT NOT NULL REFERENCES teams(team_code),
            age INTEGER
        );
        CREATE TABLE stats (
            stat_id INTEGER PRIMARY KEY,
            player_id INTEGER NOT NULL REFERENCES players(player_id),
            points INTEGER
        );
        """
    )
    conn.execute("INSERT INTO teams VALUES ('OKC', 'Oklahoma City Thunder')")
    for player_id, (name, points) in enumerate(
        [("Shai Gilgeous-Alexander", 2485), ("Chet Holmgren", 1200), ("Jalen Williams", 1400)],
        start=1,
    ):
        conn.execute(
            "INSERT INTO players (player_id, player_name, team_code, age) VALUES (?, ?, 'OKC', 25)",
            (player_id, name),
        )
        conn.execute("INSERT INTO stats (player_id, points) VALUES (?, ?)", (player_id, points))
    conn.commit()
    conn.close()
    return path


@pytest.fixture
def sql_db(nba_db, monkeypatch):
    """Le SQL Tool vise la base du projet : on la remplace par la base temporaire."""
    monkeypatch.setattr(sql_tool, "DB_FILE", nba_db)
    return nba_db


def _fake_generate(decision):
    """Fabrique un faux `generate_sql` renvoyant une décision figée (aucun appel API)."""
    def _fake(question, schema_description=None):
        return decision

    return _fake


# --- 4. Pipeline : exécution sûre + refus, toujours via le SQL Tool ------------

def test_pipeline_executes_safe_generated_sql(sql_db, monkeypatch):
    """Une requête générée SÛRE est validée puis exécutée via le SQL Tool sécurisé."""
    decision = LlmSqlDecision(
        should_query=True,
        sql="SELECT p.player_name, s.points FROM stats s "
            "JOIN players p ON p.player_id = s.player_id ORDER BY s.points DESC LIMIT 5",
        reason="classement des marqueurs",
        expected_result_type="classement",
    )
    monkeypatch.setattr(pipeline, "generate_sql", _fake_generate(decision))
    result = pipeline.run_llm_sql("Quel joueur a marqué le plus de points ?")
    assert result.validation_status == "ok"
    assert result.execution_status == "ok"
    assert result.row_count == 3
    assert "Shai Gilgeous-Alexander" in result.answer


def test_llm_answer_has_no_technical_metadata(sql_db, monkeypatch):
    """La réponse llm_sql ne contient AUCUNE phrase méta ; les chiffres et contextes restent.

    But : comparaison RAGAS équitable avec le mode contrôlé (la faithfulness ne doit pas
    être pénalisée par une phrase technique non vérifiable contre les contextes).
    """
    decision = LlmSqlDecision(
        should_query=True,
        sql="SELECT p.player_name, s.points FROM stats s "
            "JOIN players p ON p.player_id = s.player_id ORDER BY s.points DESC LIMIT 1",
        reason="le plus de points",
        expected_result_type="valeur unique",
    )
    monkeypatch.setattr(pipeline, "generate_sql", _fake_generate(decision))
    result = pipeline.run_llm_sql("Quel joueur a marqué le plus de points ?")

    lowered = result.answer.lower()
    for forbidden in ("sql généré", "généré par le llm", "lecture seule", "résultat généré", "mode llm"):
        assert forbidden not in lowered, f"Phrase méta encore présente : {forbidden}"
    # Les faits chiffrés restent dans la réponse.
    assert "Shai Gilgeous-Alexander" in result.answer
    assert "2485" in result.answer
    # Libellés FR plutôt que noms de colonnes bruts.
    assert "Joueur" in result.answer and "player_name" not in result.answer
    # Contextes présents et cohérents : chaque fait de la réponse est vérifiable.
    contexts = result.context_lines()
    assert contexts and "2485" in " ".join(contexts)
    assert "player_name" not in " ".join(contexts)


def test_prompt_requests_ranking_top_list():
    """Le prompt demande un tri (ORDER BY) pour présenter un classement, pas une seule ligne."""
    from utils.sql.llm_sql_generator import LLM_SQL_SYSTEM_PROMPT

    prompt = LLM_SQL_SYSTEM_PROMPT.lower()
    assert "order by" in prompt
    assert "classement" in prompt or "superlatif" in prompt


def test_prompt_contains_fewshot_examples():
    """Le prompt système contient bien le bloc few-shot et ses exemples clés."""
    from utils.sql.llm_sql_generator import LLM_SQL_FEWSHOT_EXAMPLES, LLM_SQL_SYSTEM_PROMPT

    # Le bloc séparé est effectivement injecté dans le prompt envoyé au LLM.
    assert LLM_SQL_FEWSHOT_EXAMPLES in LLM_SQL_SYSTEM_PROMPT

    prompt = LLM_SQL_SYSTEM_PROMPT
    # Exemples positifs clés.
    assert "three_points_attempted >= 100" in prompt          # 3P% avec filtre de volume
    assert "points > 1000" in prompt and "LAL" in prompt      # Lakers + seuil
    # Exemples de refus (should_query=false, sql null).
    assert "should_query=false" in prompt
    assert "domicile" in prompt                                # refus domicile/extérieur
    assert "salaire" in prompt                                 # refus hors schéma
    assert "meilleur joueur" in prompt                         # refus ambiguïté


def test_fewshot_sql_examples_pass_assert_read_only():
    """Chaque requête d'exemple passe assert_read_only ; les refus n'ont pas de requête."""
    from utils.sql.llm_sql_generator import _FEWSHOT_EXAMPLES
    from utils.sql.sql_tool import assert_read_only

    for example in _FEWSHOT_EXAMPLES:
        if example["should_query"]:
            assert example["sql"] is not None
            assert_read_only(example["sql"])  # ne doit pas lever
        else:
            assert example["sql"] is None     # refus -> sql null


def test_llm_ranking_answer_shows_top5_even_if_llm_limits_to_one(tmp_path, monkeypatch):
    """Même si le LLM met `LIMIT 1`, la réponse affiche un top 5 (on retire ce LIMIT).

    Réponse = top 5 (comme le contrôlé) ; contextes = toutes les lignes récupérées."""
    path = str(tmp_path / "nba_rank.sqlite")
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE teams (team_code TEXT PRIMARY KEY, team_name TEXT NOT NULL);
        CREATE TABLE players (player_id INTEGER PRIMARY KEY, player_name TEXT NOT NULL UNIQUE,
                              team_code TEXT NOT NULL REFERENCES teams(team_code));
        CREATE TABLE stats (stat_id INTEGER PRIMARY KEY, player_id INTEGER NOT NULL, points INTEGER);
        """
    )
    conn.execute("INSERT INTO teams VALUES ('OKC', 'Oklahoma City Thunder')")
    for pid in range(1, 8):  # 7 joueurs
        conn.execute("INSERT INTO players VALUES (?, ?, 'OKC')", (pid, f"Joueur Numero {pid}"))
        conn.execute("INSERT INTO stats VALUES (?, ?, ?)", (pid, pid, pid * 100))
    conn.commit()
    conn.close()
    monkeypatch.setattr(sql_tool, "DB_FILE", path)

    decision = LlmSqlDecision(
        should_query=True,
        # Le LLM se limite à UNE ligne : le pipeline doit retirer ce LIMIT et présenter un top.
        sql="SELECT p.player_name, s.points FROM stats s "
            "JOIN players p ON p.player_id = s.player_id ORDER BY s.points DESC LIMIT 1",
        reason="le plus de points",
        expected_result_type="valeur unique",
    )
    monkeypatch.setattr(pipeline, "generate_sql", _fake_generate(decision))
    result = pipeline.run_llm_sql("Quel joueur a marqué le plus de points ?")

    # Le `LIMIT 1` du LLM est retiré -> on récupère le classement complet (7 lignes).
    assert result.row_count == 7
    # La réponse affiche un top 5 (5 lignes numérotées), comme le mode contrôlé.
    assert result.answer.startswith("1. ")
    assert "5. " in result.answer and "6. " not in result.answer
    assert result.answer.count("\n") == 4  # 5 lignes -> 4 sauts de ligne
    # Les contextes couvrent TOUTES les lignes récupérées (7), pour la vérification RAGAS.
    assert len(result.context_lines()) == 7


def test_pipeline_refuses_generated_write_query_without_executing(sql_db, monkeypatch):
    """Une requête dangereuse GÉNÉRÉE est refusée à la validation, jamais exécutée."""
    decision = LlmSqlDecision(should_query=True, sql="DROP TABLE players", reason="(simulé)")
    monkeypatch.setattr(pipeline, "generate_sql", _fake_generate(decision))
    result = pipeline.run_llm_sql("Supprime la table players.")
    assert result.validation_status == "refused"
    assert result.execution_status == "skipped"
    assert result.row_count is None
    # La table n'a pas été touchée : la base est intacte.
    conn = sqlite3.connect(sql_db)
    assert conn.execute("SELECT COUNT(*) FROM players").fetchone()[0] == 3
    conn.close()


def test_pipeline_refuses_multiple_statements(sql_db, monkeypatch):
    """Deux instructions enchaînées (SELECT puis DROP) sont refusées : une seule requête est autorisée."""
    decision = LlmSqlDecision(
        should_query=True, sql="SELECT 1; DROP TABLE players", reason="(simulé)"
    )
    monkeypatch.setattr(pipeline, "generate_sql", _fake_generate(decision))
    result = pipeline.run_llm_sql("…")
    assert result.validation_status == "refused"
    assert result.execution_status == "skipped"


def test_pipeline_handles_llm_refusal(sql_db, monkeypatch):
    """Quand le LLM refuse (should_query=False), rien n'est validé ni exécuté."""
    decision = LlmSqlDecision(should_query=False, sql=None, reason="hors périmètre")
    monkeypatch.setattr(pipeline, "generate_sql", _fake_generate(decision))
    result = pipeline.run_llm_sql("Quelle est la météo à Paris ?")
    assert result.should_query is False
    assert result.generation_status == "ok"  # refus assumé : la génération a réussi
    assert result.validation_status == "skipped"
    assert result.execution_status == "skipped"


def test_pipeline_traces_generation_error_coherently(sql_db, monkeypatch):
    """Un échec de génération (API, sortie non conforme) est tracé en GÉNÉRATION.

    Il ne doit JAMAIS être rangé sous execution_* (record incohérent), ni confondu
    avec un refus assumé du LLM : `generation_status` le distingue explicitement.
    """
    def _raise(question, schema_description=None):
        raise RuntimeError("API indisponible")

    monkeypatch.setattr(pipeline, "generate_sql", _raise)
    result = pipeline.run_llm_sql("Quel joueur a marqué le plus de points ?")
    assert result.generation_status == "error"
    assert result.generation_error  # message tracé (sans secret)
    # Cohérence : aucune erreur rangée sous une étape non atteinte.
    assert result.execution_status == "skipped"
    assert result.execution_error is None
    assert result.validation_status == "skipped"
    record = result.as_record()
    assert record["generation_status"] == "error"
    assert record["execution_error"] == ""


# --- 5. Compatibilité du routeur (mode contrôlé vs llm) -----------------------

def test_router_controlled_mode_does_not_use_llm(sql_db, monkeypatch):
    """En mode contrôlé (défaut), la route SQL n'appelle jamais le pipeline LLM."""
    def _boom(*args, **kwargs):
        raise AssertionError("Le mode contrôlé ne doit pas appeler run_llm_sql.")

    monkeypatch.setattr(pipeline, "run_llm_sql", _boom)
    # SQL_GENERATION_MODE reste "controlled" : le mapping figé répond.
    result = router.answer_question("Quel joueur a marqué le plus de points ?", manager=None)
    assert result.route == "sql"
    assert "Shai Gilgeous-Alexander" in result.answer


def test_router_llm_mode_routes_through_secure_tool(sql_db, monkeypatch):
    """En mode llm, la route SQL exécute la requête générée via le SQL Tool sécurisé."""
    monkeypatch.setattr(router, "SQL_GENERATION_MODE", "llm")
    decision = LlmSqlDecision(
        should_query=True,
        sql="SELECT p.player_name, s.points FROM stats s "
            "JOIN players p ON p.player_id = s.player_id ORDER BY s.points DESC LIMIT 5",
        reason="classement",
        expected_result_type="classement",
    )
    monkeypatch.setattr(pipeline, "generate_sql", _fake_generate(decision))
    result = router.answer_question("Quel joueur a marqué le plus de points ?", manager=None)
    assert result.route == "sql"
    assert result.mode == "llm_sql"
    assert "Shai Gilgeous-Alexander" in result.answer
    assert result.retrieved_contexts  # contextes = lignes du résultat


def test_router_llm_mode_refuses_dangerous_generated_sql(sql_db, monkeypatch):
    """En mode llm, une requête dangereuse générée est refusée et la base reste intacte.

    La question est une question chiffrée NBA légitime (routée vers SQL) ; on simule un
    LLM qui produirait malgré tout une requête destructrice : elle doit être bloquée.
    """
    monkeypatch.setattr(router, "SQL_GENERATION_MODE", "llm")
    decision = LlmSqlDecision(should_query=True, sql="DROP TABLE players", reason="(simulé)")
    monkeypatch.setattr(pipeline, "generate_sql", _fake_generate(decision))
    result = router.answer_question("Quel joueur a marqué le plus de points ?", manager=None)
    assert result.route == "sql"
    assert result.mode == "llm_sql"
    assert result.notice  # message d'information (aucun chiffre inventé)
    conn = sqlite3.connect(sql_db)
    assert conn.execute("SELECT COUNT(*) FROM players").fetchone()[0] == 3
    conn.close()


def test_llm_mode_uses_season_fallback_without_calling_llm(sql_db, monkeypatch):
    """En mode llm, une question domicile/extérieur passe par le repli saison clair,
    sans solliciter le générateur LLM (réponse identique au mode contrôlé)."""
    monkeypatch.setattr(router, "SQL_GENERATION_MODE", "llm")

    def _boom(*args, **kwargs):
        raise AssertionError("Le repli saison ne doit pas appeler le LLM.")

    monkeypatch.setattr(pipeline, "run_llm_sql", _boom)
    result = router.answer_question(
        "Compare les points à domicile et à l'extérieur des équipes.", manager=None
    )
    assert result.route == "sql"
    assert "match par match" in result.answer           # absence de granularité signalée
    assert result.retrieved_contexts                    # agrégat saison réel (points par équipe)


# --- 6. Non-régression : evaluate_ragas.py et signature du routeur ------------

def test_evaluate_ragas_still_compiles():
    """Le script d'évaluation RAGAS reste syntaxiquement valide (pas de casse)."""
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts", "evaluate_ragas.py")
    py_compile.compile(path, doraise=True)


def test_answer_question_keeps_force_route_param():
    """L'API du routeur utilisée par evaluate_ragas reste compatible (force_route)."""
    params = inspect.signature(router.answer_question).parameters
    assert "force_route" in params
    assert "manager" in params
