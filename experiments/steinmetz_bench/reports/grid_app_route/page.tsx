import { promises as fs } from "fs"
import path from "path"

// Minimal scaffold for a GATED grid-app route. Drop this directory into
// grid-app `app/app/<route>/` (behind the existing auth gate) and replace the <pre>
// with the project's markdown renderer if one exists. The whitepaper markdown
// (STEINMETZ_WHITEPAPER.md) is colocated so the route is self-contained.
export default async function SteinmetzWhitepaperPage() {
  const file = path.join(process.cwd(), "app", "app", "whitepaper", "STEINMETZ_WHITEPAPER.md")
  const markdown = await fs.readFile(file, "utf8")
  return (
    <div className="h-full overflow-y-auto">
      <div className="max-w-3xl mx-auto w-full px-6 py-8">
        <pre className="whitespace-pre-wrap font-mono text-sm">{markdown}</pre>
      </div>
    </div>
  )
}
