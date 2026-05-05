import type { Metadata } from "next"
import ProjectGrid from "@/components/ProjectGrid"
import { getAllProjects } from "@/lib/content"

export const metadata: Metadata = {
  title: "Projects | Ahmad Nayfeh",
  description: "A collection of work, side projects, and experiments.",
}

export default async function ProjectsPage() {
  const projects = await getAllProjects()

  return (
    <div className="mx-auto w-full max-w-[1400px] px-6 pb-24 pt-12 md:px-10 lg:px-16">
      <header className="mb-14 border-b border-border pb-8">
        <div className="mb-5 flex items-center gap-3 animate-fade-up">
          <span aria-hidden className="h-px w-8 bg-accent" />
          <span className="font-mono text-[11px] uppercase tracking-[0.22em] text-muted-foreground">
            Selected Work · {projects.length} project{projects.length === 1 ? "" : "s"}
          </span>
        </div>
        <div className="grid grid-cols-12 items-end gap-x-6 gap-y-4">
          <h1
            className="col-span-12 font-display text-display-xl text-balance lg:col-span-8 animate-fade-up [animation-delay:60ms]"
            style={{ animationFillMode: "both" }}
          >
            Things I built, broke, and learned from
          </h1>
          <p
            className="col-span-12 max-w-md text-base text-muted-foreground lg:col-span-4 animate-fade-up [animation-delay:120ms]"
            style={{ animationFillMode: "both" }}
          >
            Each project is here because it taught me something I still use —
            from end-to-end ML systems to small experiments in signal
            processing and applied AI.
          </p>
        </div>
      </header>

      <ProjectGrid projects={projects} />
    </div>
  )
}
