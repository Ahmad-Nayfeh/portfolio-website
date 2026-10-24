import type { Route } from "next"
import Link from "next/link"
import { ArrowRight, Mail, Linkedin, Github, FileDown } from "lucide-react"
import Hero from "@/components/home/Hero"
import WorkSection from "@/components/home/WorkSection"
import SectionHeader from "@/components/drawing/SectionHeader"
import ProjectCard from "@/components/ProjectCard"
import BlogCard from "@/components/BlogCard"
import { getAllProjects, getHomeContent, getRecentPosts, getWorkContent } from "@/lib/content"
import { siteConfig, tierOf } from "@/config/site"

export default async function HomePage() {
  const [home, work, projects, posts] = await Promise.all([
    getHomeContent(),
    getWorkContent(),
    getAllProjects(),
    getRecentPosts(3),
  ])

  // Selected work: the apps I built and released, newest first.
  const selected = projects.filter((p) => tierOf(p.category) === "products").slice(0, 3)

  return (
    <>
      <Hero content={home} />

      <section id="work" className="scroll-mt-24 border-y border-border bg-card/40">
        <div className="mx-auto w-full max-w-[1240px] px-5 py-20 md:px-8 md:py-28">
          <SectionHeader
            kicker="Selected work"
            title="Apps I designed, built, and released."
            description="Each one is in real use, with releases and maintenance by me."
            action={
              <Link href={"/projects" as Route} className="redline inline-flex items-center gap-2 font-mono text-[12px] uppercase tracking-[0.14em]">
                All projects <ArrowRight size={13} />
              </Link>
            }
          />
          <div className="grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-3">
            {selected.map((p) => (
              <ProjectCard key={p.slug} project={p} />
            ))}
          </div>
        </div>
      </section>

      {work && <WorkSection work={work} />}

      {posts.length > 0 && (
        <section id="notebook" className="border-y border-border bg-card/40">
          <div className="mx-auto w-full max-w-[1240px] px-5 py-20 md:px-8 md:py-28">
            <SectionHeader
              kicker="Notebook"
              title="Experiments, written up."
              description="Articles with runnable code and measured results, on signals, images, and machine learning."
              action={
                <Link href={"/blog" as Route} className="redline inline-flex items-center gap-2 font-mono text-[12px] uppercase tracking-[0.14em]">
                  All entries <ArrowRight size={13} />
                </Link>
              }
            />
            <div className="border-t-[1.5px] border-foreground">
              {posts.map((post, i) => (
                <BlogCard key={post.slug} post={post} index={i} />
              ))}
            </div>
          </div>
        </section>
      )}

      <section id="contact" className="mx-auto w-full max-w-[1240px] scroll-mt-24 px-5 py-20 md:px-8 md:py-28">
        <SectionHeader kicker="Contact" title="Let's talk." />
        <div className="grid grid-cols-1 gap-8 lg:grid-cols-12">
          <p className="text-[1.15rem] leading-relaxed text-foreground/85 lg:col-span-5">
            Open to conversations about switchgear engineering, design automation, and building software products. Email is the fastest way to reach me.
          </p>
          <div className="lg:col-span-7">
            <a
              href={`mailto:${siteConfig.contact.email}`}
              className="group flex items-center justify-between gap-4 border-[1.5px] border-foreground bg-card px-5 py-5 transition-colors hover:border-accent md:px-7 md:py-6"
            >
              <span className="flex items-center gap-4">
                <Mail size={20} className="text-accent" />
                <span className="text-[15px] font-semibold tracking-tight [overflow-wrap:anywhere] sm:text-lg md:text-2xl">{siteConfig.contact.email}</span>
              </span>
              <ArrowRight size={20} className="shrink-0 transition-transform group-hover:translate-x-1" />
            </a>
            <div className="mt-3 grid grid-cols-1 gap-3 sm:grid-cols-3">
              <ContactLink href={siteConfig.links.linkedin} icon={<Linkedin size={15} />} label="LinkedIn" />
              <ContactLink href={siteConfig.links.github} icon={<Github size={15} />} label="GitHub" />
              <ContactLink href={siteConfig.resume} icon={<FileDown size={15} />} label="Download CV" />
            </div>
          </div>
        </div>
      </section>
    </>
  )
}

function ContactLink({ href, icon, label }: { href: string; icon: React.ReactNode; label: string }) {
  return (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      className="flex items-center justify-between border border-border bg-card px-4 py-3 text-[14px] font-medium transition-colors hover:border-foreground"
    >
      <span className="flex items-center gap-2.5 text-muted-foreground">
        {icon}
        <span className="text-foreground">{label}</span>
      </span>
      <ArrowRight size={14} className="text-muted-foreground" />
    </a>
  )
}
