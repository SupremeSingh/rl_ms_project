import path from 'node:path';
import { pathToFileURL } from 'node:url';

const workspaceDir = '/Users/manmit/Desktop/rl_ms_project';
const skillDir = '/Users/manmit/.codex/plugins/cache/openai-primary-runtime/presentations/26.905.11957/skills/presentations';
const buildDir = path.join(workspaceDir, '.codex-build/rl_project_deck');
const candidatePath = path.join(buildDir, 'candidate.pptx');
const finalPath = path.join(workspaceDir, 'artifacts/math_rl_project_briefing.pptx');

const { finalizePresentation } = await import(pathToFileURL(
  path.join(skillDir, 'container_tools/artifact_tool_utils.mjs'),
).href);

const result = await finalizePresentation({
  explicitTotalSlideCount: 10,
  requiredNativeTableOwnerSlides: [],
  requiredNativeChartOwnerSlides: [],
  workspaceDir,
  candidatePath,
  finalPath,
  pythonExecutable: '/Users/manmit/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3',
  integrityValidatorPath: path.join(skillDir, 'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath: path.join(skillDir, 'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs: [
    '--expected-slide-size-emu', '12192000,6858000',
    '--validate-bullet-geometry',
    '--validate-heading-fit',
  ],
  fontPolicy: { basis: 'design', families: ['Helvetica Neue'] },
  verifyArtifactToolImport: true,
  receiptPath: path.join(buildDir, 'math_rl_project_briefing.validation.json'),
});
console.log(JSON.stringify(result, null, 2));
