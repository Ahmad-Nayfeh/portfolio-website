import type { Route } from "next"
import Image from "next/image"
import Link from "next/link"
import { ArrowRight, ArrowUpRight } from "lucide-react"
import type { Project } from "@/types"
import { cn } from "@/lib/utils"
import { hasLink, splitTitle } from "@/components/projects/projectName"

interface ProductFeatureProps {
  project: Project
  index: number
  reverse?: boolean
}

/** Large product panel: write-up on one side, screenshots on a drawing sheet on the other. */
export default function ProductFeature({ project, index, reverse = false }: ProductFeatureProps) {
  const { name, tagline } = splitTitle(project.title)
  const shots = project.gallery.slice(0, 3)
  const sheetNo = `P-${String(index + 1).padStart(2, "0")}`
  const href = `/projects/${project.slug}` as Route

  return (
    <article className="grid grid-cols-1 items-center gap-x-12 gap-y-8 lg:grid-cols-12">
      <div className={cn("lg:col-span-5", reverse && "lg:order-2")}>
        <div className="flex items-center gap-3">
          <span className="font-mono text-[11px] font-semibold text-accent">{sheetNo}</span>
          {project.status && (
            <span className="inline-flex items-center gap-2">
              <span className="lamp" aria-hidden />
              <span className="label">{project.status}</span>
            </span>
          )}
        </div>

        <h3 className="mt-4 font-display text-[clamp(2rem,4vw,3rem)] leading-none">
          <Link href={href} className="redline">
            {name}
          </Link>
        </h3>
        {tagline && <p className="mt-3 text-lg font-medium text-foreground/80">{tagline}</p>}
        <p className="mt-5 text-[16px] leading-relaxed text-muted-foreground">{project.excerpt}</p>

        {project.highlights.length > 0 && (
          <ul className="mt-6 border-t border-border">
            {project.highlights.map((h, i) => (
              <li key={h} className="flex gap-4 border-b border-border py-2.5 text-[14.5px] leading-snug">
                <span className="mt-[2px] font-mono text-[10.5px] text-muted-foreground">{String(i + 1).padStart(2, "0")}</span>
                <span className="text-foreground/90">{h}</span>
              </li>
            ))}
          </ul>
        )}

        <div className="mt-5 flex flex-wrap gap-1.5">
          {project.tags.map((t) => (
            <span key={t} className="chip">
              {t}
            </span>
          ))}
        </div>

        <div className="mt-7 flex flex-wrap gap-3">
          <Link href={href} className="btn btn-primary">
            Case study <ArrowRight size={14} />
          </Link>
          {hasLink(project.liveDemoUrl) && (
            <a href={project.liveDemoUrl} target="_blank" rel="noopener noreferrer" className="btn btn-ghost">
              {project.linkLabel ?? "Open live"} <ArrowUpRight size={14} />
            </a>
          )}
        </div>
      </div>

      <div className={cn("lg:col-span-7", reverse && "lg:order-1")}>
        <Link href={href} className="sheet sheet-hover group block overflow-hidden" aria-label={`${name} case study`}>
          <div className="flex items-center justify-between border-b border-border px-4 py-2">
            <span className="label text-[9.5px]">{sheetNo} · {name}</span>
            <span className="label text-[9.5px]">{shots.length > 0 ? "Screens" : "Overview"}</span>
          </div>
          {shots.length > 0 ? (
            <div className="relative flex items-end justify-center gap-3 px-4 pb-0 pt-8 sm:gap-5 md:px-8 md:pt-10">
              {shots.map((src, i) => (
                <div
                  key={src}
                  className={cn(
                    "phone w-[30%] max-w-[210px] transition-transform duration-300",
                    i === 1 ? "z-10 -mb-0 translate-y-0 group-hover:-translate-y-2" : "translate-y-6 group-hover:translate-y-4",
                  )}
                >
                  <Image src={src} alt={`${name} screen ${i + 1}`} width={540} height={1140} sizes="(max-width: 768px) 30vw, 210px" />
                </div>
              ))}
            </div>
          ) : (
            <div className="relative aspect-[16/9]">
              <Image src={project.coverImage} alt={name} fill className="object-cover" sizes="(max-width: 1024px) 100vw, 720px" />
            </div>
          )}
        </Link>
      </div>
    </article>
  )
}
