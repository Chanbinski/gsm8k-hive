"""GSM8K solver — the artifact agents evolve.

Takes a math word problem on stdin, prints the numeric answer on stdout.
"""

import sys
import os
import re

from openai import OpenAI


def solve(question: str) -> str:
    """Solve a GSM8K math problem. Return the numeric answer as a string."""
    client = OpenAI()

    response = client.chat.completions.create(
        model=os.environ.get("SOLVER_MODEL", "gpt-4.1-nano"),
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a math tutor solving a word problem step by step.\n"
                    "1. Read the problem carefully.\n"
                    "2. Identify the quantities and what is being asked.\n"
                    "3. Show your work step by step, one arithmetic operation at a time.\n"
                    "4. After your steps, write the final answer on its own line in "
                    "exactly this format:\n"
                    "#### <number>\n"
                    "The number must be a plain integer or decimal (no commas, no units, "
                    "no dollar signs)."
                ),
            },
            {"role": "user", "content": question},
        ],
        temperature=0,
        max_tokens=512,
    )

    text = response.choices[0].message.content.strip()

    # Try to extract answer after #### delimiter (GSM8K convention)
    match = re.search(r'####\s*(-?\d[\d,]*\.?\d*)', text)
    if match:
        return match.group(1).replace(',', '')

    # Fallback: take the last number in the response
    numbers = re.findall(r'-?\d[\d,]*\.?\d*', text)
    if numbers:
        return numbers[-1].replace(',', '')

    return text


if __name__ == "__main__":
    question = sys.stdin.read().strip()
    print(solve(question))
