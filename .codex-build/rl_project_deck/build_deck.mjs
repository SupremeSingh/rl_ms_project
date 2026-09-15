import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { FileBlob, PresentationFile } from '@oai/artifact-tool';

const workspaceDir = '/Users/manmit/Desktop/rl_ms_project';
const buildDir = path.join(workspaceDir, '.codex-build/rl_project_deck');
const skillDir = '/Users/manmit/.codex/plugins/cache/openai-primary-runtime/presentations/26.905.11957/skills/presentations';
const templatePath = '/Users/manmit/.codex/plugins/cache/openai-curated-remote/openai-templates/0.1.1/skills/artifact-template-simple-light-mode/assets/reference.pptx';
const draftPath = path.join(buildDir, 'candidate.pptx');

const { resolvePresentationFont } = await import(pathToFileURL(path.join(skillDir, 'container_tools/artifact_tool_utils.mjs')).href);
const fontFamily = resolvePresentationFont();

const presentation = await PresentationFile.importPptx(await FileBlob.load(templatePath));
while (presentation.slides.count > 0) presentation.slides.remove(0);

const W = 1280;
const H = 720;
const C = {
  ink: '#111111',
  muted: '#5E6470',
  faint: '#8B919C',
  rule: '#D9DDE3',
  blue: '#2D5BFF',
  bluePale: '#EEF2FF',
  coral: '#E45B52',
  coralPale: '#FFF0EE',
  green: '#167A64',
  greenPale: '#EAF7F3',
  white: '#FFFFFF',
};
const blankLayout = '/ppt/slideLayouts/slideLayout8.xml';
const titleLayout = '/ppt/slideLayouts/slideLayout1.xml';

function box(slide, x, y, w, h, fill = 'none', line = 'none', width = 0, geometry = 'rect') {
  return slide.shapes.add({ geometry, position: { left: x, top: y, width: w, height: h }, fill, line: { fill: line, width } });
}
function text(slide, value, x, y, w, h, style = {}) {
  const shape = slide.shapes.add({ geometry: 'textbox', position: { left: x, top: y, width: w, height: h }, fill: 'none', line: { fill: 'none', width: 0 } });
  shape.text = value;
  shape.text.style = { typeface: fontFamily, fontSize: 20, color: C.ink, autoFit: 'none', ...style };
  return shape;
}
function rule(slide, x, y, w, color = C.rule, h = 1) { box(slide, x, y, w, h, color, 'none', 0, 'rect'); }
function label(slide, n, section = 'MATH RL PROJECT') {
  text(slide, section, 40, 28, 300, 22, { fontSize: 13, bold: true, color: C.muted, letterSpacing: 1.2 });
  text(slide, String(n).padStart(2, '0'), 1185, 28, 55, 22, { fontSize: 13, bold: true, color: C.faint, alignment: 'right' });
}
function title(slide, value, n, section) {
  label(slide, n, section);
  text(slide, value, 40, 72, 1160, 54, { fontSize: 34, bold: true, color: C.ink });
}
function note(slide, value) { slide.speakerNotes.text = value; }
function addBody(slide, value, x, y, w, h, color = C.ink, size = 20, opts = {}) {
  return text(slide, value, x, y, w, h, { fontSize: size, color, ...opts });
}
function addRow(slide, y, name, body, accent = C.blue, x = 72, w = 1100) {
  box(slide, x, y + 4, 6, 44, accent, 'none');
  text(slide, name, x + 24, y, 235, 26, { fontSize: 18, bold: true, color: C.ink });
  text(slide, body, x + 275, y, w - 275, 52, { fontSize: 18, color: C.muted });
}

// 1. Cover
{
  const s = presentation.slides.add({ layoutId: titleLayout });
  s.background.fill = C.white;
  text(s, 'MATH RL PROJECT', 40, 32, 300, 22, { fontSize: 13, bold: true, color: C.muted, letterSpacing: 1.2 });
  text(s, 'Learning with the\nmodel’s own internal state', 40, 150, 920, 150, { fontSize: 48, bold: true, color: C.ink });
  text(s, 'PPO, GRPO and lightweight token-level critics for verified math', 44, 335, 850, 34, { fontSize: 22, color: C.muted });
  rule(s, 44, 414, 180, C.blue, 4);
  text(s, 'Qwen2.5-Math-1.5B  ·  GSM8K  ·  Math-Verify  ·  VERL  ·  Duke GPU cluster', 44, 465, 930, 28, { fontSize: 16, color: C.muted });
  text(s, 'Project briefing', 44, 612, 200, 20, { fontSize: 14, color: C.faint });
  note(s, 'Source: project codebase and experiment plan. Model: Qwen/Qwen2.5-Math-1.5B base checkpoint, not the instruct variant.');
}

// 2. Goal
{
  const s = presentation.slides.add({ layoutId: blankLayout });
  s.background.fill = C.white;
  title(s, 'Research question', 2);
  addBody(s, 'Can a language model supply the information needed for its own inexpensive critic?', 72, 160, 1120, 70, C.ink, 30, { bold: true });
  addBody(s, 'We test whether a small value head on the actor’s existing hidden states can guide PPO as well as a conventional critic, at lower fitting and memory cost.', 72, 250, 1030, 78, C.muted, 22);
  rule(s, 72, 374, 1120, C.rule, 1);
  addRow(s, 420, 'Actor', 'The LLM writes a solution one token at a time.', C.blue);
  addRow(s, 484, 'Critic', 'A value estimate asks: “How likely is this unfinished solution to succeed?”', C.coral);
  addRow(s, 548, 'Test', 'Compare semi-gradient TD and LSTD under matched data, features and compute.', C.green);
  note(s, 'This is a hypothesis test. A conventional critic can still win; that outcome is informative.');
}

// 3. Papers
{
  const s = presentation.slides.add({ layoutId: blankLayout });
  s.background.fill = C.white;
  title(s, 'What prior work contributes', 3, 'PRIOR WORK');
  const x0 = 72, x1 = 330, x2 = 920;
  text(s, 'Paper', x0, 150, 235, 24, { fontSize: 14, bold: true, color: C.faint });
  text(s, 'Lesson used here', x1, 150, 560, 24, { fontSize: 14, bold: true, color: C.faint });
  rule(s, 72, 182, 1120, C.rule, 1);
  const rows = [
    ['PPO Algorithms\nSchulman et al., 2017', 'Clipped probability ratios support several updates on one sampled batch.'],
    ['Your Language Model is Its Own Critic\nChoi et al., 2026 · POISE', 'Actor hidden states can support a low-cost value baseline for verifiable rewards.'],
    ['Least-Squares Policy Iteration\nLagoudakis & Parr, 2003', 'Least-squares Bellman evaluation motivates an LSTD critic solver.'],
    ['Shallow value updates\nProject synthesis', 'Keep the deep representation fixed while testing cheap, controlled head fitting.'],
  ];
  rows.forEach((r, i) => {
    const y = 208 + i * 92;
    text(s, r[0], x0, y, 235, 62, { fontSize: 17, bold: true, color: C.ink });
    text(s, r[1], x1, y, 780, 56, { fontSize: 18, color: C.muted });
    rule(s, 72, y + 72, 1120, C.rule, 1);
  });
  note(s, 'Sources: /Users/manmit/Downloads/1707.06347v2.pdf (PPO Algorithms); /Users/manmit/Downloads/2605.07579v2.pdf (Your Language Model is Its Own Critic / POISE); project plan notes for the LSTD and shallow-update framing.');
}

// 4. RL setup
{
  const s = presentation.slides.add({ layoutId: blankLayout });
  s.background.fill = C.white;
  title(s, 'The RL problem in one page', 4, 'RL SETUP');
  box(s, 72, 148, 530, 410, C.bluePale, 'none');
  text(s, 'One generated solution', 104, 178, 440, 30, { fontSize: 20, bold: true, color: C.blue });
  addBody(s, 'state  sₜ = (question, tokens before t)\naction  aₜ = next token\nterminal reward  rₜ = 1 if Math-Verify says correct, else 0', 104, 245, 440, 150, C.ink, 23);
  addBody(s, 'The reward arrives at the end. Every earlier token needs a credit signal.', 104, 446, 430, 65, C.muted, 19);
  box(s, 650, 148, 542, 410, C.coralPale, 'none');
  text(s, 'Value before the next token', 682, 178, 460, 30, { fontSize: 20, bold: true, color: C.coral });
  addBody(s, 'Vπ(sₜ) = E[r_terminal | sₜ]', 682, 250, 460, 46, C.ink, 28, { bold: true });
  addBody(s, 'The critic predicts the chance that the current prefix will eventually earn reward 1 under the current actor.', 682, 335, 450, 90, C.muted, 21);
  addBody(s, 'Causal rule: read φθ(sₜ) before choosing aₜ.', 682, 470, 450, 46, C.ink, 20, { bold: true });
  note(s, 'The value target is a success probability under the binary terminal reward, although an unconstrained linear head may predict outside [0, 1].');
}

// 5. Architecture
{
  const s = presentation.slides.add({ layoutId: blankLayout });
  s.background.fill = C.white;
  title(s, 'System architecture', 5, 'SYSTEM');
  const y = 248, h = 160;
  const nodes = [
    { x: 50, w: 200, head: 'Slurm', body: 'Allocates\nA5000 GPUs', fill: C.bluePale, accent: C.blue },
    { x: 285, w: 220, head: 'Container', body: 'Pins CUDA,\nPyTorch and VERL', fill: '#F5F5F5', accent: C.ink },
    { x: 540, w: 220, head: 'VERL + Ray', body: 'Runs the RL\ntraining loop', fill: C.greenPale, accent: C.green },
    { x: 795, w: 190, head: 'vLLM', body: 'Generates\nrollouts', fill: C.bluePale, accent: C.blue },
    { x: 1015, w: 215, head: 'Math-Verify', body: 'Returns\n0 or 1 reward', fill: C.coralPale, accent: C.coral },
  ];
  nodes.forEach((n, i) => {
    box(s, n.x, y, n.w, h, n.fill, n.accent, 1, 'roundRect');
    text(s, n.head, n.x + 20, y + 24, n.w - 40, 28, { fontSize: 20, bold: true, color: n.accent });
    text(s, n.body, n.x + 20, y + 72, n.w - 40, 60, { fontSize: 19, color: C.ink });
    if (i < nodes.length - 1) rule(s, n.x + n.w + 10, y + h / 2, nodes[i + 1].x - (n.x + n.w) - 20, C.rule, 2);
  });
  rule(s, 72, 492, 1120, C.rule, 1);
  addBody(s, 'PyTorch + FSDP update the actor and PPO critic. Checkpoints, rollouts and reports remain separate for PPO and GRPO.', 72, 528, 1120, 48, C.muted, 19);
  addBody(s, 'One combined Slurm job: two GPUs for PPO, two GPUs for GRPO.', 72, 600, 1120, 32, C.ink, 18, { bold: true });
  note(s, 'The architecture is an editable diagram. Slurm handles allocation; the container provides the pinned runtime; VERL owns the algorithmic loop; our code supplies prompts, rewards, configuration, launch orchestration and audits.');
}

// 6. PPO and GRPO mechanics
{
  const s = presentation.slides.add({ layoutId: blankLayout });
  s.background.fill = C.white;
  title(s, 'Two policy-gradient baselines', 6, 'ALGORITHMS');
  box(s, 72, 150, 530, 420, C.bluePale, 'none');
  text(s, 'PPO + GAE', 104, 180, 430, 32, { fontSize: 24, bold: true, color: C.blue });
  addBody(s, 'δₜ = rₜ + γV(sₜ₊₁) − V(sₜ)\nAₜ = Σₗ (γλ)ˡ δₜ₊ₗ\nρₜ = πθ(aₜ|sₜ) / πold(aₜ|sₜ)\nL = min(ρₜAₜ, clip(ρₜ)Aₜ)', 104, 250, 440, 190, C.ink, 23);
  addBody(s, 'One answer per question in this pilot. A separate critic predicts values and GAE spreads the terminal signal backward.', 104, 486, 440, 56, C.muted, 18);
  box(s, 650, 150, 542, 420, C.coralPale, 'none');
  text(s, 'GRPO', 682, 180, 450, 32, { fontSize: 24, bold: true, color: C.coral });
  addBody(s, 'Aᵢ = (rᵢ − mean(r₁…rₙ))\n      / (std(r₁…rₙ) + ε)', 682, 260, 460, 100, C.ink, 26);
  addBody(s, 'Four answers per question. Answers above their group mean get positive advantage; below-mean answers get negative advantage.', 682, 405, 460, 90, C.muted, 20);
  addBody(s, 'No critic and no GAE in this baseline.', 682, 515, 460, 30, C.ink, 18, { bold: true });
  note(s, 'Both methods use the same actor, prompt, reward contract and clipped policy update. Their variance-reduction mechanisms differ.');
}

// 7. Lightweight critic
{
  const s = presentation.slides.add({ layoutId: blankLayout });
  s.background.fill = C.white;
  title(s, 'The lightweight critic experiment', 7, 'CRITIC');
  addBody(s, 'Reuse the actor representation. Learn only a small head.', 72, 150, 1100, 42, C.ink, 27, { bold: true });
  box(s, 72, 225, 360, 250, C.bluePale, 'none');
  text(s, 'Actor features', 104, 255, 290, 28, { fontSize: 20, bold: true, color: C.blue });
  addBody(s, 'φθ(sₜ)\n= hidden state of the prefix\nbefore action aₜ', 104, 320, 290, 105, C.ink, 24);
  rule(s, 454, 350, 88, C.rule, 2);
  box(s, 565, 225, 295, 250, C.greenPale, 'none');
  text(s, 'Small value head', 595, 255, 240, 28, { fontSize: 20, bold: true, color: C.green });
  addBody(s, 'V̂(sₜ) = wᵀφθ(sₜ)', 595, 335, 240, 42, C.ink, 26, { bold: true });
  addBody(s, 'Only w changes\nduring fitting', 595, 410, 220, 52, C.muted, 18);
  rule(s, 885, 350, 88, C.rule, 2);
  box(s, 995, 225, 197, 250, C.coralPale, 'none');
  text(s, 'Compare solvers', 1020, 255, 150, 28, { fontSize: 20, bold: true, color: C.coral });
  addBody(s, 'TD(0)\nLSTD(0)', 1020, 335, 145, 75, C.ink, 24, { bold: true });
  rule(s, 72, 520, 1120, C.rule, 1);
  addBody(s, 'Semi-gradient TD:  w ← w + α δₜ φθ(sₜ), with the next-state prediction detached.', 72, 554, 1120, 32, C.ink, 19);
  addBody(s, 'LSTD: solve the regularized batch equations  (A + ηI)w = b.', 72, 598, 1120, 32, C.ink, 19);
  note(s, 'The initial comparison has fresh trajectories, no cross-policy replay, causal pre-token features, matched features/data/regularization/initialization and equal-data plus equal-fitting-time controls.');
}

// 8. Protocol
{
  const s = presentation.slides.add({ layoutId: blankLayout });
  s.background.fill = C.white;
  title(s, 'Experimental protocol', 8, 'PLAN');
  const items = [
    ['1', 'Reward audit', 'Use Math-Verify; separate wrong answers, parse failures and truncation.'],
    ['2', 'PPO reference', 'Validate conventional PPO/GAE, critic updates and checkpoint evidence.'],
    ['3', 'Frozen actor', 'Hold Qwen fixed; compare TD-small, TD-large, LSTD and return regression.'],
    ['4', 'Online PPO', 'Connect each lightweight critic to fresh-policy PPO.'],
    ['5', 'Final comparison', 'Use ≥3 seeds; report accuracy per generated token and GPU-hour.'],
  ];
  items.forEach((it, i) => {
    const y = 150 + i * 88;
    text(s, it[0], 72, y, 40, 38, { fontSize: 27, bold: true, color: i < 2 ? C.blue : C.coral });
    text(s, it[1], 142, y + 3, 250, 30, { fontSize: 21, bold: true, color: C.ink });
    text(s, it[2], 410, y + 3, 730, 38, { fontSize: 19, color: C.muted });
    rule(s, 72, y + 58, 1120, C.rule, 1);
  });
  addBody(s, 'Controls: prompt-level cross-fitting, no initial cross-policy replay, detached critic fitting, matched regularization, and honest compute accounting.', 72, 620, 1120, 36, C.ink, 18, { bold: true });
  note(s, 'If return regression predicts well but TD(0) does not, test matched TD(λ) and LSTD(λ) before concluding token-level value learning fails. Critic λ is distinct from GAE λ.');
}

// 9. Work done
{
  const s = presentation.slides.add({ layoutId: blankLayout });
  s.background.fill = C.white;
  title(s, 'Work completed', 9, 'STATUS');
  const cols = [72, 385, 700, 1015];
  const headers = ['Milestone', 'Evidence', 'Result', 'Interpretation'];
  headers.forEach((h, i) => text(s, h, cols[i], 150, i === 0 ? 285 : 260, 24, { fontSize: 14, bold: true, color: C.faint }));
  rule(s, 72, 182, 1120, C.rule, 1);
  const rows = [
    ['Stage 1 audit', '200 responses\n10 reviewed', '100% agreement\non reviewed set', 'Audit still incomplete'],
    ['PPO pilot', '10 updates\n160 train responses', '31/32 validation\ninitial and final', 'Execution passed'],
    ['GRPO pilot', '10 updates\n320 train responses', '31/32 validation\ninitial and final', 'Execution passed'],
    ['Infrastructure', 'Duke A5000\n4 GPUs total', 'Actor gradients,\ncheckpoints, reports', 'Learning gain unproven'],
  ];
  rows.forEach((r, i) => {
    const y = 212 + i * 90;
    r.forEach((v, j) => text(s, v, cols[j], y, j === 0 ? 285 : 260, 58, { fontSize: 18, color: j === 2 ? C.ink : C.muted, bold: j === 0 || j === 2 }));
    rule(s, 72, y + 68, 1120, C.rule, 1);
  });
  addBody(s, 'Pilot reports prove that the machinery runs. They do not prove improved math accuracy or a useful critic.', 72, 610, 1120, 34, C.coral, 19, { bold: true });
  note(s, 'Evidence from Duke job 12589388: PPO report and GRPO report both returned execution_check_pass=true; validation reward mean@1 remained 0.96875 at steps 0, 5 and 10. Stage 1 review report remains development-only and incomplete.');
}

// 10. Next steps
{
  const s = presentation.slides.add({ layoutId: blankLayout });
  s.background.fill = C.white;
  title(s, 'Next steps', 10, 'NEXT');
  addBody(s, '1  Evaluate base, PPO and GRPO on 500 held-out questions.', 72, 160, 1120, 38, C.ink, 23, { bold: true });
  addBody(s, 'Same prompts, greedy decoding and Math-Verify. Report wrong to right and right to wrong changes.', 98, 204, 1060, 32, C.muted, 18);
  addBody(s, '2  Freeze the actor and benchmark the critics.', 72, 280, 1120, 38, C.ink, 23, { bold: true });
  addBody(s, 'Compare TD-small, TD-large, LSTD and return regression on held-out prefixes and continuations.', 98, 324, 1060, 32, C.muted, 18);
  addBody(s, '3  Run short online PPO pilots with the best heads.', 72, 400, 1120, 38, C.ink, 23, { bold: true });
  addBody(s, 'Keep data fresh and measure whether value quality translates into policy improvement.', 98, 444, 1060, 32, C.muted, 18);
  rule(s, 72, 520, 1120, C.rule, 1);
  addBody(s, 'Deliverable', 72, 560, 170, 28, C.blue, 19, { bold: true });
  addBody(s, 'A validated accuracy–compute comparison among lightweight critics, with conventional PPO as the reference.', 265, 560, 880, 36, C.ink, 20, { bold: true });
  note(s, 'The immediate next experiment is frozen-actor critic evaluation. A frozen-actor win does not guarantee online PPO success; the online pilots test that link directly.');
}

await fs.mkdir(buildDir, { recursive: true });
await (await PresentationFile.exportPptx(presentation)).save(draftPath);
const montage = await presentation.export({ format: 'webp', montage: true, scale: 1 });
await fs.writeFile(path.join(buildDir, 'montage.webp'), new Uint8Array(await montage.arrayBuffer()));
for (let i = 0; i < presentation.slides.items.length; i++) {
  const slide = presentation.slides.getItem(i);
  const preview = await presentation.export({ slide, format: 'png', scale: 1 });
  await fs.writeFile(path.join(buildDir, `slide-${i + 1}.png`), new Uint8Array(await preview.arrayBuffer()));
}
console.log(JSON.stringify({ draftPath, slides: presentation.slides.items.length, fontFamily }, null, 2));
