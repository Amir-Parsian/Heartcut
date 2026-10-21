"""Heartcut Studio: the whole pipeline as a no-code, drag-and-drop web app.

``heartcut studio`` starts it in the browser. A person drops in a spreadsheet of
messages they have already sorted, drags its columns onto *message* and *category*,
picks a teacher LLM and the students to compare, and gets the benchmark, a try-it box,
a way to label a whole spreadsheet and a downloadable model, without writing a config
file or a line of code.

Under the hood every studio project is an ordinary experiment (``experiment.yaml`` in
the project folder) run by the same stage functions as the CLI, so anything made in the
studio can be reproduced or extended with ``heartcut run -c <project>/experiment.yaml``.
"""
