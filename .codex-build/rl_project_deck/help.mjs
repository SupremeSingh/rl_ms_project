import { FileBlob, PresentationFile } from '@oai/artifact-tool';
const src='/Users/manmit/.codex/plugins/cache/openai-curated-remote/openai-templates/0.1.1/skills/artifact-template-simple-light-mode/assets/reference.pptx';
const p=await PresentationFile.importPptx(await FileBlob.load(src));
for (const q of ['slide remove delete','slide text style','shape add','speaker notes','presentation slides collection']) {
 console.log('QUERY',q); console.log(await p.help(q,{search:q,include:['index','examples','notes'],maxChars:8000}));
}
