"""The classic theatre: a perfect-information dyad of one human and one agent.

A **classic theatre** is a theatre of exactly two participants, one of which is
a human. The journal is **totally ordered** -- append-only, no fork, no join,
no rewind.  Both participants have shared access to the complete history (the
journal), so the interaction is a *perfect-information dyad*: the information
channel is complete by construction.  Recall imperfections -- human forgetting,
LLM context compression -- are properties of the participants at the
implementation layer, not of the theatre's information structure.

This concept encompasses **chat bots** and **coding agents** (human-in-the-loop
with a single LLM).  It does **not** encompass multi-agent systems (debate /
critic theatres), fully automated processes (batch / always-on theatres with no
human), group chats (concurrent theatre), or search (branching state).  Those
are different composition rules over the same participants and journal, and they
are future siblings of this package, not replacements for it.

Although the structure of the theatre is the same, we provide two example theatres
for two different user interfaces - ACP for graphical UIs and also a simple REPL
for kicking the tyres from the commandline.

"""
