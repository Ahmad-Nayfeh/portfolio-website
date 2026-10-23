export const siteConfig = {
  name: "Ahmad Nayfeh",
  title: "Ahmad Nayfeh — Electrical Design Engineer & AI Automation",
  role: "Electrical Design Engineer",
  description:
    "Electrical design engineer in medium-voltage switchgear who builds AI-assisted tools and ships complete software products.",
  url: "https://ahmadnayfeh.vercel.app",
  location: "Riyadh, Saudi Arabia",
  resume: "/resume.pdf",
  links: {
    github: "https://github.com/Ahmad-Nayfeh",
    linkedin: "https://www.linkedin.com/in/ahmad-nayfeh2000/",
  },
  contact: {
    email: "ahmadnayfeh2000@gmail.com",
    location: "Riyadh, Saudi Arabia",
  },
  nav: [
    { name: "Index", path: "/", sheet: "01" },
    { name: "Projects", path: "/projects", sheet: "02" },
    { name: "Notebook", path: "/blog", sheet: "03" },
    { name: "About", path: "/about", sheet: "04" },
  ],
} as const

/**
 * Project tiers. A project's `category` frontmatter decides its tier.
 * Anything not listed (e.g. pipeline-generated "AI Implementation" pages)
 * falls into the Lab tier.
 */
export const projectTiers = [
  {
    id: "products",
    name: "Products",
    categories: ["Products"],
    blurb: "Complete applications with real users, releases, and maintenance.",
  },
  {
    id: "research",
    name: "Research & ML",
    categories: ["Research"],
    blurb: "Deep learning and signal processing work from my M.Sc. and independent study.",
  },
  {
    id: "lab",
    name: "Lab",
    categories: [] as string[],
    blurb: "Smaller tools and experiments, kept for reference.",
  },
] as const

export type TierId = (typeof projectTiers)[number]["id"]

export function tierOf(category: string): TierId {
  for (const tier of projectTiers) {
    if ((tier.categories as readonly string[]).includes(category)) return tier.id
  }
  return "lab"
}
