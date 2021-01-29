#!/usr/bin/env python3
"""
Script that takes in input from the user with the prompt Q:
and prints A: as a response. If the user inputs exit, quit,
goodbye, or bye, case insensitive, print A: Goodbye and exit.
"""
exits = ["exit", "quit", "goodbye", "bye"]
while True:
    question = input("Q: ").lower().strip()
    if question in exits:
        print("A: Goodbye")
        exit()
    else:
        print("A:")
