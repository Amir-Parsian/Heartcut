# Heartcut Studio: no code, drag and drop

Heartcut Studio runs the whole Heartcut pipeline in your web browser. You drop in a
spreadsheet, drag its columns into place and pick models from cards. You don't write
any config files or commands beyond the one that opens the Studio. It runs on your own
computer, and your data stays there unless you choose a hosted teacher such as Claude.

## Install and open it

You need Python 3.11 or newer. In a terminal:

```bash
pip install "heartcut[studio,teachers]"
heartcut studio
```

Your browser opens at <http://127.0.0.1:8765>. Leave the terminal window open while
you use the Studio, and press Ctrl+C in it when you are done. Your projects are saved in
a `heartcut-studio` folder inside the folder you started from. Use
`heartcut studio --workspace <folder>` to keep them somewhere else.

From a clone of this repository you can run `make studio` instead.

| Extra | What it adds |
|---|---|
| `studio` | The Studio itself, the Tiny student, and reading Excel files |
| `teachers` | Claude and local (Ollama) teachers. Without it, only the practice run is available |
| `students,export` | The Small and Medium students (PyTorch). Large download; optional |

## The five steps

1. **Data.** Drag a spreadsheet onto the drop zone, or click it to pick a file. CSV,
   Excel (`.xlsx`), TSV and JSON lines files work, up to 50 MB. Each row is one message,
   and one column says which category the message belongs to. No data yet? Use the 120
   sample IT-helpdesk tickets.
2. **Columns.** The Studio guesses which column holds the message and which holds the
   category. Drag column cards onto **Message** and **Category** to change its guess, or
   click a card to choose from a menu. If you drop several columns on Message, such as a
   subject and a body, they are joined into one message. The Studio counts the examples
   in each category and warns you about anything that would spoil the results, for
   example categories with fewer than 5 examples.
3. **Teacher.** The teacher is the large AI that labels your messages for the students
   to learn from.
   - **Practice run**: free and instant. A stand-in teacher copies your own categories
     with deliberate mistakes, so you can try every step at no cost. Its scores are for
     practice only.
   - **Claude Haiku, Sonnet or Opus**: these need an [Anthropic API
     key](https://console.anthropic.com/). Paste the key into the Studio, or set
     `ANTHROPIC_API_KEY` before you start it. The key is kept in memory and is never
     written to disk. Set a **spending limit**: the teacher stops when the limit is
     reached, and you never pay twice for an answer you already have.
   - **Local model (Ollama)**: runs on your computer, so your messages never leave it.
     Install [Ollama](https://ollama.com) and run `ollama pull qwen2.5:7b-instruct` first.
4. **Students.** Drag the small models you want to compare into **Your line-up**, or use
   their Add buttons. **Tiny** trains in seconds on any computer. **Small** and
   **Medium** usually understand meaning better, but they need the `students,export`
   extra and take minutes to train. **Thorough** trains each student three times and
   draws a learning curve. It takes longer than **Quick**.
5. **Review.** Check the summary and the estimated teacher cost, then press **Start
   training**.

## While it trains

A progress list shows each stage: preparing your data, the teacher labelling,
checking the teacher's answers, training, then the final exam. You can close the tab;
training continues as long as the Studio is running. **Stop after this step** pauses
the run. **Resume** continues it and reuses everything already finished, so no
answer is paid for twice. If something goes wrong, the Studio explains what happened
and what to do next. For example, after a spending-limit stop you can raise the limit
right there and resume.

## The results

- **Your winner**: the student that agreed best with the teacher on the messages set
  aside to choose a winner. The final exam never picks the winner; it only grades it.
- **Correct answers, cost per 1,000 messages and time per message**: the winner beside
  the teacher. Each model is graded against your own categories on messages that no model
  saw while learning.
- **Try it**: type a message to see the category the winner picks, how sure it is, and
  the next best guesses.
- **Sort a whole spreadsheet**: drop a file of new messages. It comes back as a CSV
  file with `predicted_category` and `confidence` columns added. The file needs the same
  message columns as your training data, or a single column of messages.
- **Download model**: a zip file holding the trained model and a README that explains
  how to run it as a web service with `heartcut serve`.
- **Every model**, **Charts** and **Where the winner gets mixed up**: the full benchmark
  and the most common mistakes, with examples.

## For developers

Each project is an ordinary Heartcut experiment. Its folder contains:

```
heartcut-studio/projects/<project>/
    experiment.yaml     the experiment config (heartcut run -c <this file> reproduces it)
    data/               the uploaded examples, split into train.jsonl and test.jsonl
    artifacts/          stage outputs: prepared data, teacher labels, students, exports
    results/            results.json, results.md, charts, audit, model card
    run.log
```

The Studio's server is `heartcut.studio.app.create_app`, which serves a FastAPI JSON API
and a single page written in plain JavaScript, with no build step. By default it listens
only on 127.0.0.1 and turns away requests that name another host. With
`--host 0.0.0.0`, anyone who can reach your machine can use the Studio and any API key
pasted into it: there is no login.
