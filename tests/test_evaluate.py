"""Tests for SQL evaluation — verifiable without GPU."""

from __future__ import annotations

from text2sql.evaluate import exact_match, execute_sql, execution_match, valid_sql


def test_execute_sql_simple():
    schema = "CREATE TABLE users (id INTEGER, name TEXT);"
    ok, results = execute_sql("SELECT * FROM users", schema)
    assert ok
    assert results == []


def test_execute_sql_with_data():
    schema = "CREATE TABLE t (x INTEGER); INSERT INTO t VALUES (1), (2), (3);"
    ok, results = execute_sql("SELECT x FROM t WHERE x > 1", schema)
    assert ok
    assert sorted(results) == [(2,), (3,)]


def test_execute_sql_invalid():
    schema = "CREATE TABLE t (x INTEGER);"
    ok, error = execute_sql("SELEC * FROM t", schema)
    assert not ok
    assert isinstance(error, str)


def test_exact_match_identical():
    assert exact_match("SELECT * FROM users", "SELECT * FROM users")


def test_exact_match_whitespace():
    assert exact_match("SELECT  *  FROM  users", "SELECT * FROM users")


def test_exact_match_case():
    assert exact_match("select * from users", "SELECT * FROM users")


def test_exact_match_trailing_semicolon():
    assert exact_match("SELECT * FROM users;", "SELECT * FROM users")


def test_exact_match_different():
    assert not exact_match("SELECT id FROM users", "SELECT name FROM users")


def test_execution_match_same_results():
    schema = "CREATE TABLE t (x INTEGER); INSERT INTO t VALUES (1), (2);"
    assert execution_match("SELECT x FROM t", "SELECT x FROM t", schema)


def test_execution_match_equivalent_queries():
    schema = "CREATE TABLE t (a INTEGER, b INTEGER); INSERT INTO t VALUES (1, 2);"
    assert execution_match(
        "SELECT a, b FROM t WHERE a = 1",
        "SELECT a, b FROM t WHERE a = 1",
        schema,
    )


def test_execution_match_different_results():
    schema = "CREATE TABLE t (x INTEGER); INSERT INTO t VALUES (1), (2), (3);"
    assert not execution_match("SELECT x FROM t WHERE x > 1", "SELECT x FROM t WHERE x < 2", schema)


def test_valid_sql_good():
    assert valid_sql("SELECT * FROM users WHERE id = 1")


def test_valid_sql_bad():
    assert not valid_sql("")
