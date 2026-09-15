import { FileBlob, PresentationFile } from '@oai/artifact-tool';

const source = '/Users/manmit/.codex/plugins/cache/openai-curated-remote/openai-templates/0.1.1/skills/artifact-template-simple-light-mode/assets/reference.pptx';
const presentation = await PresentationFile.importPptx(await FileBlob.load(source));
console.log('slides', presentation.slides.items.length);
const snapshot = await presentation.inspect({
  kind: 'slide,textbox,shape,image,table,chart,notes,layout',
  maxChars: 30000,
});
console.log(snapshot.ndjson);
