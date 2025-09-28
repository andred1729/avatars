import argparse

from talk import talk, DEFAULT_USER_PROMPT


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Herdora and print the generated script without playback."
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        default=DEFAULT_USER_PROMPT,
        help="Optional prompt to feed Herdora; defaults to the built-in prompt.",
    )
    args = parser.parse_args()

    result = talk(args.prompt)
    print(result["text"])


if __name__ == "__main__":
    main()
