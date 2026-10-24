import type { Route } from "next"
import Link from "next/link"
import { Github, Linkedin, Mail, FileDown } from "lucide-react"
import { siteConfig } from "@/config/site"

const REV = new Intl.DateTimeFormat("en-US", { month: "short", year: "numeric" }).format(new Date())

export default function Footer() {
  const year = new Date().getFullYear()
  return (
    <footer className="mt-12 border-t-[1.5px] border-foreground bg-card/80">
      <div className="mx-auto w-full max-w-[1240px] px-5 py-10 md:px-8">
        {/* Title block */}
        <div className="grid grid-cols-2 border-[1.5px] border-foreground md:grid-cols-12 [&>div]:border-b [&>div]:border-r [&>div]:border-border">
          <div className="col-span-2 px-4 py-4 md:col-span-5 md:row-span-2">
            <div className="label text-[9.5px]">Portfolio of</div>
            <div className="mt-1.5 text-2xl font-semibold tracking-tight">{siteConfig.name}</div>
            <p className="mt-2 max-w-sm text-sm leading-relaxed text-muted-foreground">
              Electrical design engineer in medium-voltage switchgear. I build AI-assisted tools and ship software products.
            </p>
          </div>
          <div className="col-span-2 px-4 py-3 md:col-span-4">
            <div className="label text-[9.5px]">Contact</div>
            <a href={`mailto:${siteConfig.contact.email}`} className="redline mt-1 inline-flex items-center gap-2 text-[14px]">
              <Mail size={13} className="text-muted-foreground" />
              {siteConfig.contact.email}
            </a>
          </div>
          <div className="px-4 py-3 md:col-span-3">
            <div className="label text-[9.5px]">Location</div>
            <div className="mt-1 text-[14px]">{siteConfig.location}</div>
          </div>
          <div className="col-span-2 px-4 py-3 md:col-span-4">
            <div className="label text-[9.5px]">Links</div>
            <div className="mt-1.5 flex flex-wrap gap-x-5 gap-y-1 text-[14px]">
              <a href={siteConfig.links.linkedin} target="_blank" rel="noopener noreferrer" className="redline inline-flex items-center gap-1.5">
                <Linkedin size={13} className="text-muted-foreground" /> LinkedIn
              </a>
              <a href={siteConfig.links.github} target="_blank" rel="noopener noreferrer" className="redline inline-flex items-center gap-1.5">
                <Github size={13} className="text-muted-foreground" /> GitHub
              </a>
              <a href={siteConfig.resume} target="_blank" rel="noopener noreferrer" className="redline inline-flex items-center gap-1.5">
                <FileDown size={13} className="text-muted-foreground" /> CV
              </a>
            </div>
          </div>
          <div className="px-4 py-3 md:col-span-3">
            <div className="label text-[9.5px]">Updated</div>
            <div className="mt-1 font-mono text-[13px]">{REV}</div>
          </div>
          <div className="col-span-2 px-4 py-3 md:col-span-12">
            <div className="label text-[9.5px]">Pages</div>
            <nav aria-label="Footer" className="mt-1.5 flex flex-wrap gap-x-6 gap-y-1 text-[14px]">
              {siteConfig.nav.map((item) => (
                <Link key={item.path} href={item.path as Route} className="redline">
                  {item.name}
                </Link>
              ))}
            </nav>
          </div>
        </div>

        <p className="mt-5 font-mono text-[10.5px] uppercase tracking-[0.14em] text-muted-foreground">
          © {year} {siteConfig.name} · Built with Next.js · Set in IBM Plex
        </p>
      </div>
    </footer>
  )
}
