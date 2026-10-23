import type { Route } from "next"
import Link from "next/link"
import { ArrowRight, Mail, Linkedin, Github, FileDown } from "lucide-react"
import Hero from "@/components/home/Hero"
import NowPanel from "@/components/home/NowPanel"
import WorkSection from "@/components/home/WorkSection"
import SectionHeader from "@/components/drawing/SectionHeader"
import ProductFeature from "@/components/projects/ProductFeature"
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

  const products = projects.filter((p) => tierOf(p.category) === "products")
  const flagship = products.filter((p) => p.featured).slice(0, 2)
  const otherProducts = products.filter((p) => !flagship.includes(p))
  const research = projects.filter((p) => tierOf(p.category) === "research").slice(0, 3)

  return (
    <>
      <Hero content={home} />
      <NowPanel items={home.now} />

      {work && <WorkSection work={work} />}

      {/* 02 · Products */}
      <section id="products" className="scroll-mt-24 border-y border-border bg-card/40">
        <div className="mx-auto w-full max-w-[1240px] px-5 py-20 md:px-8 md:py-28">
          <SectionHeader
            number="02"
            kicker="Products"
            title="Software I designed, built, and ship."
            description="End-to-end products with real users: backend, web, and mobile, released and maintained by me."
          />
          <div className="space-y-20 md:space-y-28">
            {flagship.map((p, i) => (
              <ProductFeature key={p.slug} project={p} index={i} reverse={i % 2 === 1} />
            ))}
          </div>
          {otherProducts.length > 0 && (
            <div className="mt-16 border-t border-border pt-8">
              <div className="label mb-4">Also shipped</div>
              <div className={otherProducts.length === 1 ? "" : "grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-3"}>
                {otherProducts.map((p, i) => (
                  <ProjectCard
                    key={p.slug}
                    project={p}
                    layout={otherProducts.length === 1 ? "wide" : "stacked"}
                    sheetNo={`P-${String(flagship.length + i + 1).padStart(2, "0")}`}
                  />
                ))}
              </div>
            </div>
          )}
        </div>
      </section>

      {/* 03 · Research */}
      {research.length > 0 && (
        <section id="research" className="mx-auto w-full max-w-[1240px] scroll-mt-24 px-5 py-20 md:px-8 md:py-28">
          <SectionHeader
            number="03"
            kicker="Research & ML"
            title="Deep learning and signal processing."
            description="From my M.Sc. at KFUPM and independent study: medical imaging, predictive maintenance, and self-supervised learning."
            action={
              <Link href={"/projects" as Route} className="redline inline-flex items-center gap-2 font-mono text-[12px] uppercase tracking-[0.14em]">
                All projects <ArrowRight size={13} />
              </Link>
            }
          />
          <div className="grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-3">
            {research.map((p, i) => (
              <ProjectCard key={p.slug} project={p} sheetNo={`R-${String(i + 1).padStart(2, "0")}`} />
            ))}
          </div>
        </section>
      )}

      {/* 04 · Notebook */}
      {posts.length > 0 && (
        <section id="notebook" className="border-y border-border bg-card/40">
          <div className="mx-auto w-full max-w-[1240px] px-5 py-20 md:px-8 md:py-28">
            <SectionHeader
              number="04"
              kicker="Notebook"
              title="Writing, to understand things properly."
              description="Long-form explainers on signals, images, and how machines learn."
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

      {/* 05 · Contact */}
      <section id="contact" className="mx-auto w-full max-w-[1240px] scroll-mt-24 px-5 py-20 md:px-8 md:py-28">
        <SectionHeader number="05" kicker="Contact" title="Let's talk." />
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
