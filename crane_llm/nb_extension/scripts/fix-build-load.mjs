// Repair `jupyterlab._build.load` in the generated labextension/package.json.
//
// @jupyter/builder 1.x locates the freshly written remoteEntry bundle with
// `glob.sync(path.join(staticPath, 'remoteEntry.*.js'))`. On Windows
// `path.join` yields backslashes, and glob 9 and later treat a backslash as an
// escape character rather than a separator, so the pattern matches nothing.
// The builder then records an empty file name and writes
// `load: path.join('static', '')`, which is just `"static"`.
//
// JupyterLab resolves that to a directory rather than a module, so the
// extension is listed but never loads in the browser. This step puts the real
// entry point back. It is a no-op on platforms where the builder got it right.

import { readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const outputDir = process.argv[2] ?? 'labextension';
const packagePath = join(outputDir, 'package.json');
const staticDir = join(outputDir, 'static');

const entries = readdirSync(staticDir).filter(name =>
  /^remoteEntry\..+\.js$/.test(name)
);

if (entries.length !== 1) {
  console.error(
    `[fix-build-load] expected exactly one remoteEntry bundle in ${staticDir}, found ${entries.length}: ${entries.join(', ')}`
  );
  process.exit(1);
}

const pkg = JSON.parse(readFileSync(packagePath, 'utf8'));
const build = pkg.jupyterlab?._build;

if (!build) {
  console.error(`[fix-build-load] ${packagePath} has no jupyterlab._build block`);
  process.exit(1);
}

// Always a URL path, so forward slashes regardless of platform.
const expected = `static/${entries[0]}`;

if (build.load === expected) {
  process.exit(0);
}

const previous = build.load;
build.load = expected;
writeFileSync(packagePath, `${JSON.stringify(pkg, null, 2)}\n`);
console.log(
  `[fix-build-load] corrected load: ${JSON.stringify(previous)} -> ${JSON.stringify(expected)}`
);
