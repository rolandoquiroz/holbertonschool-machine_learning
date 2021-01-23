#!/usr/bin/env python3
"""
Script that takes in input from the user with the prompt Q:
and prints A: as a response. If the user inputs exit, quit,
goodbye, or bye, case insensitive, print A: Goodbye and exit.
"""
question = ""
while not (question == "exit" or question == "quit" or question == "goodbye"
           or question == "bye"):
    print("Q: ", end="")
    question = input().lower()
    print("A:")
else:
    print("A: Goodbye")
    exit()
        