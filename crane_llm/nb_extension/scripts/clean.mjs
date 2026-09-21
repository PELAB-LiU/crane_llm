// Delete build outputs. Replaces the `rimraf` devDependency, which pulled in
// `glob` through a floating range and broke the build when glob 13.0.6 was
// published with an empty dist/commonjs directory. Node's own fs.rmSync does
// the same job with nothing to resolve.
import { rmSync } from 'node:fs';

for (const target of process.argv.slice(2)) {
  rmSync(target, { recursive: true, force: true });
}
