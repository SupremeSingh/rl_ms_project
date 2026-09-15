import { FileBlob, PresentationFile } from '@oai/artifact-tool';
const p=await PresentationFile.importPptx(await FileBlob.load('/Users/manmit/.codex/plugins/cache/openai-curated-remote/openai-templates/0.1.1/skills/artifact-template-simple-light-mode/assets/reference.pptx'));
console.log('before',p.slides.count);
try { const out=p.slides.remove(0); console.log('ret',out,'after',p.slides.count); } catch(e) { console.error(e); }
