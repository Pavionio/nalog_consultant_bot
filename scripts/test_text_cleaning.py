from __future__ import annotations

import unittest
from types import SimpleNamespace

from src.rag.text_cleaning import build_corpus_boilerplate_stats, clean_text


class TextCleaningTests(unittest.TestCase):
    def test_clean_basic_removes_entities_and_normalizes_spaces(self) -> None:
        source = "Текст&nbsp;&nbsp;с  пробелами &amp; сущностями\r\n\r\nИ второй абзац."
        result = clean_text(source, profile="clean_basic")
        self.assertNotIn("&nbsp;", result.text)
        self.assertNotIn("  ", result.text)
        self.assertIn("Текст с пробелами & сущностями", result.text)
        self.assertIn("И второй абзац.", result.text)

    def test_clean_legal_keeps_legal_lines_and_removes_navigation(self) -> None:
        source = (
            "Версия для печати\n"
            "Письмо ФНС России от 01.01.2024 № АБ-1-2/3\n"
            "статья 346.43 НК РФ\n"
            "Поделиться\n"
        )
        result = clean_text(source, profile="clean_legal")
        self.assertNotIn("Версия для печати", result.text)
        self.assertNotIn("Поделиться", result.text)
        self.assertIn("Письмо ФНС России от 01.01.2024 № АБ-1-2/3", result.text)
        self.assertIn("статья 346.43 НК РФ", result.text)

    def test_clean_aggressive_removes_repeated_noise_but_keeps_legal_markers(self) -> None:
        source = (
            "Главная > Налоги > Раздел\n"
            ".... !!!\n"
            "шумовая строка\n"
            "шумовая строка\n"
            "НДФЛ уплачивается налоговым агентом\n"
            "НК РФ\n"
            "пункт 1 статьи 1\n"
            "Письмо от 12.03.2024 № 03-04-05/12345\n"
        )
        result = clean_text(source, profile="clean_aggressive")
        self.assertNotIn("шумовая строка\nшумовая строка", result.text)
        self.assertIn("НДФЛ", result.text)
        self.assertIn("НК РФ", result.text)
        self.assertIn("пункт 1", result.text)
        self.assertIn("№ 03-04-05/12345", result.text)

    def test_clean_no_boilerplate_removes_common_service_lines_keeps_legal(self) -> None:
        legal_line = "Письмо ФНС России от 01.01.2024 № АБ-1-2/3"
        docs = [
            SimpleNamespace(text=f"Поделиться\n{legal_line}\nУникальный текст 1"),
            SimpleNamespace(text=f"Поделиться\n{legal_line}\nУникальный текст 2"),
            SimpleNamespace(text=f"Поделиться\n{legal_line}\nУникальный текст 3"),
            SimpleNamespace(text=f"Поделиться\n{legal_line}\nУникальный текст 4"),
            SimpleNamespace(text=f"Поделиться\n{legal_line}\nУникальный текст 5"),
            SimpleNamespace(text=f"Поделиться\n{legal_line}\nУникальный текст 6"),
            SimpleNamespace(text=f"Поделиться\n{legal_line}\nУникальный текст 7"),
            SimpleNamespace(text=f"Поделиться\n{legal_line}\nУникальный текст 8"),
            SimpleNamespace(text=f"Поделиться\n{legal_line}\nУникальный текст 9"),
            SimpleNamespace(text=f"Поделиться\n{legal_line}\nУникальный текст 10"),
            SimpleNamespace(text=f"Поделиться\n{legal_line}\nУникальный текст 11"),
            SimpleNamespace(text=f"Поделиться\n{legal_line}\nУникальный текст 12"),
            SimpleNamespace(text=f"Поделиться\n{legal_line}\nУникальный текст 13"),
            SimpleNamespace(text=f"Поделиться\n{legal_line}\nУникальный текст 14"),
            SimpleNamespace(text=f"Поделиться\n{legal_line}\nУникальный текст 15"),
            SimpleNamespace(text=f"Поделиться\n{legal_line}\nУникальный текст 16"),
            SimpleNamespace(text=f"Поделиться\n{legal_line}\nУникальный текст 17"),
            SimpleNamespace(text=f"Поделиться\n{legal_line}\nУникальный текст 18"),
            SimpleNamespace(text=f"Поделиться\n{legal_line}\nУникальный текст 19"),
            SimpleNamespace(text=f"Поделиться\n{legal_line}\nУникальный текст 20"),
        ]
        corpus_stats = build_corpus_boilerplate_stats(docs)
        result = clean_text(
            f"Поделиться\n{legal_line}\nОсновной текст",
            profile="clean_no_boilerplate",
            corpus_stats=corpus_stats,
        )
        self.assertNotIn("Поделиться", result.text)
        self.assertIn(legal_line, result.text)


if __name__ == "__main__":
    unittest.main()
