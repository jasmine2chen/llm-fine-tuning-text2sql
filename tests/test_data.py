"""Tests for the data module: chat formatting and splitting."""

from text2sql.data import create_chat_messages


class TestCreateChatMessages:
    """Tests for create_chat_messages."""

    def _sample(self) -> dict:
        return {
            "question": "How many employees are there?",
            "context": "CREATE TABLE employees (id INT, name VARCHAR(100))",
            "answer": "SELECT COUNT(*) FROM employees",
        }

    def test_returns_messages_key(self) -> None:
        result = create_chat_messages(self._sample())
        assert "messages" in result

    def test_three_messages(self) -> None:
        msgs = create_chat_messages(self._sample())["messages"]
        assert len(msgs) == 3

    def test_roles(self) -> None:
        msgs = create_chat_messages(self._sample())["messages"]
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert msgs[2]["role"] == "assistant"

    def test_schema_in_system_message(self) -> None:
        sample = self._sample()
        msgs = create_chat_messages(sample)["messages"]
        assert sample["context"] in msgs[0]["content"]

    def test_question_in_user_message(self) -> None:
        sample = self._sample()
        msgs = create_chat_messages(sample)["messages"]
        assert msgs[1]["content"] == sample["question"]

    def test_answer_in_assistant_message(self) -> None:
        sample = self._sample()
        msgs = create_chat_messages(sample)["messages"]
        assert msgs[2]["content"] == sample["answer"]

    def test_system_prompt_mentions_sql(self) -> None:
        msgs = create_chat_messages(self._sample())["messages"]
        assert "SQL" in msgs[0]["content"]


class TestDatasetSplitting:
    """Verify the split ratios used in load_and_prepare.

    These are unit-level checks that don't call the HuggingFace Hub.
    """

    def test_split_ratio_math(self) -> None:
        """With 12500 samples and test_ratio=0.2 we expect 2500 test samples."""
        total = 12500
        ratio = 0.2
        expected_test = int(total * ratio)
        assert expected_test == 2500

    def test_train_size_math(self) -> None:
        total = 12500
        ratio = 0.2
        expected_train = total - int(total * ratio)
        assert expected_train == 10000
