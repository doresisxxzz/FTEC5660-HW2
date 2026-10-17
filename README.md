# FTEC5660 Homework 2: CV Verification Agent

Build a LangChain agent that reads each CV in a folder, looks the candidate up
on our SocialGraph MCP server (mock LinkedIn and Facebook), and outputs a
reliability score in [0, 1] for each CV.

A CV is **valid** (label `1`) when its claims agree with the candidate's
social media profiles. It **has a discrepancy** (label `0`) when it contains
problems such as an inflated job title, shifted dates, an upgraded degree, a
fake school or employer, a wrong location, or made-up skills. Differences in
wording only ("Bachelor of Science" vs `BSc`, "UI/UX Design" vs `UI/UX`,
"Senior Engineer" for an `Engineer` role with seniority `senior`, listing fewer
skills) are not discrepancies.

## Student task

Fill in the two functions in `hw2.py` that contain `### YOUR CODE HERE`. You
may add imports, constants and helper functions above them, but do not change
the provided code below them:

- `build_agent(tools)` creates your agent from the MCP tools.
- `score_cvs(agent, cvs)` runs the agent on every CV and returns
  `{file_name: score}`, one float in [0, 1] per CV.

A score above `0.5` means "valid"; `0.5` or below means "has discrepancy". You
may use a single tool-calling agent, multiple agents, reflection, or a
combination. Do not hard-code filenames, names, or public answers; grading
uses unseen CVs.

## MCP server

The server is hosted at `https://ftec5660.ngrok.app/mcp`. `hw2.py` already
connects to it and passes you these tools:

| Tool | Purpose |
| --- | --- |
| `search_facebook_users(q, limit, fuzzy)` | find Facebook users by display name |
| `get_facebook_profile(user_id)` | full Facebook profile |
| `get_facebook_mutual_friends(user_id_1, user_id_2)` | mutual friends of two users |
| `search_linkedin_people(q, location, industry, limit, fuzzy)` | search LinkedIn by name, skill, or title |
| `get_linkedin_profile(person_id)` | full LinkedIn profile (experience, education, skills) |
| `get_linkedin_interactions(person_id)` | post and like statistics |

Print `tool.name`, `tool.description`, and `tool.args` for the full schemas.

## Setup and public test

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
echo "DEEPSEEK_API_KEY=your_key_here" > .env
python3 hw2.py --cv-folder public_test
```

Use your own DeepSeek API key (from https://platform.deepseek.com).

The program creates `results.csv` in the current directory. Its columns are
`cv`, `score`, and `correctness`. The public labels are in
`public_test/ground_truth.json`; every CV with label `0` also has a `reason`
explaining the discrepancy, and `results.csv` shows it when your agent misses
one. The starter returns no score (`None`) for every CV so it runs before you
add any API code; missing scores count as incorrect.

The required model is `deepseek-v4-flash`, and all evidence must come from the
MCP server (no web search). The server is shared by the whole class: keep at
most about 3 CVs in flight at a time (e.g. `asyncio.Semaphore(3)`). We grade
with our own API key, so never put a key in your code; `.env` must stay out of
git.


## Task 2: Fool the Verifier

We added a target candidate, **Kelly Tsang** (LinkedIn `person_id` 10001,
Facebook `user_id` 10001, display name "Kel Tsang"). Her true CV is
`task2/target_cv.pdf`.

Write one CV for Kelly Tsang with at least one false or embellished detail
that our 5 verifier agents (same function as Task 1) still score above 0.5.
Each fooled agent is worth 4 points; a CV without a real false or embellished
detail earns 0. Attack your own Task 1 agent first:

```bash
cp my_attack.pdf task2/adversarial_cv.pdf
python3 hw2.py --cv-folder task2
```

`results.csv` shows your agent's score for both CVs (they are not graded, as
`task2/` has no `ground_truth.json`). Only the PDF is the attack surface; do not
attack the MCP server. See the homework description for the full rules.

## Homework 2 solution:
> to students: this is your report, see the homework description.

### Task 1

I built a LangChain agent using the required `deepseek-v4-flash` model and the
SocialGraph MCP tools. For each CV, it searches for the candidate, checks the
identity against location and background details, retrieves the LinkedIn
profile, and compares the CV's name, city, jobs, education, and listed skills.
It treats equivalent wording as a match, but gives a low score when even one
material claim differs. The CV and tool results are treated as data rather than
instructions. One CV runs at a time, and a shared lock also serializes actual
MCP tool calls when the model requests several tools at once. Read-only tool
calls retry transient connection failures individually; each CV also has its
own retry and error handling, with an uncertain fallback score of 0.5 if
verification still fails. The latest full run of the revised implementation
classified all 7 public-test CVs correctly (7/7): CVs 1, 2, 3, and 6 scored
0.95, and CVs 4, 5, and 7 scored 0.05. This result covers the public test;
performance on unseen CVs has not been measured. A fresh clone with a newly
created virtual environment and reinstalled dependencies also scored 7/7.

### Task 2

I queried Kelly Tsang's LinkedIn and Facebook profiles (ID 10001) and confirmed
that her current role is Manager at AIA, starting in 2022. My one-page
`task2/adversarial_cv.pdf` visibly promotes this role to **Director**, while
retaining the other factual claims. It exploits a mismatch between the PDF's
rendered appearance and its extracted text: an image displays "Director, AIA"
over the underlying PDF text "Manager, AIA". The homework's unmodified
MarkItDown loader extracts the underlying Manager title and does not extract
the title in the image. A verifier receiving only that text therefore sees a
CV consistent with the MCP profile.

In one local test using my unchanged Task 1 agent, the original CV scored
0.95, a control CV with an ordinary extractable Director title scored 0.05,
and the adversarial PDF scored 0.95. A separate end-to-end run with
`python hw2.py --cv-folder task2` again scored both the original and adversarial
CVs at 0.95. This comparison supports the extraction mismatch as the mechanism.
The assignment explicitly permits techniques inside the PDF and lists hidden
text and PDF text extraction as possible approaches; the false Director claim
is present in the rendered page. I have not tested the five hidden verifiers
or their three-run majority criterion. A verifier using OCR or visual
inspection could detect the false title; final eligibility and marks are
determined by the instructors.

### Task 3: Reflection on AI Paradoxes

When the terminal finally showed 7/7, I was relieved. After dealing with
connection errors and trying to understand how to run the program, seeing
all seven CVs classified correctly felt like a clear sign that things were
working. But I still asked how I could know whether the results were really
right. Comparing the CSV with the ground truth answered that question for
the public examples. Task 2 made the question much harder.

Our adversarial CV visibly changed Kelly Tsang's job from Manager to
Director, but the verifier gave it 0.95, the same score as her genuine CV.
The ordinary version with Director written as extractable text received
0.05. The difference came from the PDF: the attack displayed Director as an
image while leaving Manager underneath as text. The loader passed Manager
to the agent, and the MCP profile also said Manager. From the text available
to it, accepting the CV made sense. From the page I could see, it was accepting
an exaggerated claim.

That was the part I found most interesting. I had focused on whether the AI
could search for the right person and compare the facts carefully. Yet even
a careful comparison could miss this problem because the visible false claim
never reached the model. A verifier could follow its instructions and still
produce a result that misled the person relying on it. I would now want to
check what the system actually read before deciding whether to trust its
judgment.

The score also looked more reassuring than it should have. In our prompt,
0.95 was a value chosen for a CV that appeared consistent. It was not the
result of measuring a 95 percent probability that the document was truthful.
Seeing the same number beside both PDFs made that distinction much clearer.
The public-test result was useful evidence, but it did not tell me how the
agent would handle every kind of document.

I noticed a similar tension in my own learning. AI assistance made it
possible to build and test something that I could not yet fully explain. I
needed help understanding the functions, the terminal output, and even the
CSV file. Getting a working result was encouraging, but it also made it easy
to move on before understanding why it worked. Reading the code and checking
the results helped me notice how much understanding was still missing.

If I continued this project, I would start by showing the extracted text
beside the rendered CV so that a reviewer could spot differences. I would
also separate an unresolved check from a finding of false information: a
connection failure should not be treated as evidence against a candidate.
These changes would make the system less convenient in some cases, because
more documents might need review, but I think that is a reasonable cost when
someone's application could be affected.

I am still pleased that the agent passed the public test and that the attack
worked locally. I just see those results differently now. Before treating an
AI answer as dependable, I want to know what information it received, what
it actually checked, and whether I understand enough to question the result.
