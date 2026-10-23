import type { CSSProperties } from "react"
import type { Route } from "next"
import Link from "next/link"

/**
 * Hero single-line diagram (SLD).
 *
 * Drawn like a switchgear SLD: an incomer feeds a busbar, and four outgoing
 * feeders (each with a load-break switch and a current transformer) end in
 * terminal boxes that link to the sections of the site. Linework draws in,
 * the switch blades close, then a slow "current" runs along the conductors.
 * All motion is CSS and is disabled under prefers-reduced-motion.
 */

export interface Feeder {
  tag: string
  title: string
  lines: [string, string]
  href: string
}

interface SingleLineDiagramProps {
  feeders: Feeder[]
  source: [string, string]
}

const X = [120, 240, 360, 480]
const BUS_Y = 150

function v(len: number, delay: number): CSSProperties {
  return { ["--len" as string]: len, animationDelay: `${delay}s` } as CSSProperties
}

function FeederLink({ href, children, label }: { href: string; children: React.ReactNode; label: string }) {
  if (href.startsWith("#")) {
    return (
      <a href={href} className="sld-feeder" aria-label={label}>
        {children}
      </a>
    )
  }
  return (
    <Link href={href as Route} className="sld-feeder" aria-label={label}>
      {children}
    </Link>
  )
}

export default function SingleLineDiagram({ feeders, source }: SingleLineDiagramProps) {
  const ink = "hsl(var(--foreground))"
  const muted = "hsl(var(--muted-foreground))"
  const paper = "hsl(var(--card))"

  return (
    <svg
      viewBox="0 0 600 470"
      className="h-auto w-full select-none"
      role="group"
      aria-labelledby="sld-title"
      fontFamily="var(--font-mono), ui-monospace, monospace"
    >
      <title id="sld-title">Single-line diagram linking to the sections of this site</title>

      {/* ── Source (two-winding transformer) ───────────────────────────── */}
      <g className="sld-fade" style={{ animationDelay: "0s" }}>
        <circle cx="300" cy="34" r="17" fill="none" stroke={ink} strokeWidth="1.5" />
        <circle cx="300" cy="56" r="17" fill="none" stroke={ink} strokeWidth="1.5" />
        <text x="330" y="34" fontSize="10" fill={muted} letterSpacing="1.2">
          {source[0]}
        </text>
        <text x="330" y="49" fontSize="11" fill={ink} fontWeight="500">
          {source[1]}
        </text>
      </g>

      {/* Incomer with circuit breaker */}
      <path d="M300 73 V104" stroke={ink} strokeWidth="1.5" fill="none" className="sld-draw" style={v(31, 0.15)} />
      <rect
        x="292"
        y="104"
        width="16"
        height="16"
        fill={paper}
        stroke={ink}
        strokeWidth="1.5"
        className="sld-fade"
        style={{ animationDelay: "0.3s" }}
      />
      <text x="278" y="116" fontSize="10" fill={muted} textAnchor="end" className="sld-fade" style={{ animationDelay: "0.35s" }}>
        Q0
      </text>
      <path d="M300 120 V150" stroke={ink} strokeWidth="1.5" fill="none" className="sld-draw" style={v(30, 0.35)} />

      {/* ── Busbar ─────────────────────────────────────────────────────── */}
      <path d={`M60 ${BUS_Y} H540`} stroke={ink} strokeWidth="4" strokeLinecap="square" fill="none" className="sld-draw" style={v(480, 0.5)} />
      <text x="60" y={BUS_Y - 12} fontSize="10" fill={muted} letterSpacing="1.2" className="sld-fade" style={{ animationDelay: "0.8s" }}>
        MAIN BUSBAR
      </text>
      <g className="sld-fade" style={{ animationDelay: "2.3s" }}>
        <circle cx="520" cy={BUS_Y - 16} r="4" fill="hsl(var(--lamp))" />
        <circle cx="520" cy={BUS_Y - 16} r="8" fill="hsl(var(--lamp) / 0.18)" />
        <text x="508" y={BUS_Y - 12} fontSize="10" fill={muted} textAnchor="end" letterSpacing="1.2">
          LIVE
        </text>
      </g>

      {/* ── Feeders ────────────────────────────────────────────────────── */}
      {feeders.slice(0, 4).map((f, i) => {
        const x = X[i]
        const d0 = 0.95 + i * 0.14
        return (
          <FeederLink key={f.tag} href={f.href} label={`${f.title}: ${f.lines.join(" ")}`}>
            {/* invisible hit area */}
            <rect x={x - 58} y={BUS_Y + 4} width="116" height="300" fill="transparent" />

            <path d={`M${x} ${BUS_Y} V200`} stroke={ink} strokeWidth="1.5" fill="none" className="sld-draw sld-hl" style={v(50, d0)} />
            {/* load-break switch: fixed-contact bar, pivot, and blade */}
            <path d={`M${x - 7} 200 H${x + 7}`} stroke={ink} strokeWidth="1.5" className="sld-fade sld-hl" style={{ animationDelay: `${d0 + 0.2}s` }} />
            <line
              x1={x}
              y1="238"
              x2={x}
              y2="202"
              stroke={ink}
              strokeWidth="1.75"
              strokeLinecap="round"
              className="sld-blade sld-hl"
              style={{ animationDelay: `${d0 + 0.75}s` }}
            />
            <circle cx={x} cy="238" r="2.6" fill={ink} className="sld-fade" style={{ animationDelay: `${d0 + 0.2}s` }} />
            <text x={x - 14} y="224" fontSize="10" fill={muted} textAnchor="end" className="sld-fade" style={{ animationDelay: `${d0 + 0.3}s` }}>
              Q{i + 1}
            </text>

            <path d={`M${x} 238 V330`} stroke={ink} strokeWidth="1.5" fill="none" className="sld-draw sld-hl" style={v(92, d0 + 0.25)} />
            {/* current transformer */}
            <circle cx={x} cy="272" r="8" fill={paper} stroke={ink} strokeWidth="1.5" className="sld-fade sld-hl" style={{ animationDelay: `${d0 + 0.45}s` }} />
            <circle cx={x} cy="284" r="8" fill="none" stroke={ink} strokeWidth="1.5" className="sld-fade sld-hl" style={{ animationDelay: `${d0 + 0.45}s` }} />
            <text x={x + 14} y="282" fontSize="10" fill={muted} className="sld-fade" style={{ animationDelay: `${d0 + 0.5}s` }}>
              CT
            </text>

            {/* terminal box */}
            <g className="sld-fade" style={{ animationDelay: `${d0 + 0.85}s` }}>
              <rect x={x - 54} y="330" width="108" height="86" fill={paper} stroke={ink} strokeWidth="1.5" className="sld-hl" />
              <text x={x - 46} y="347" fontSize="9.5" fill={muted} letterSpacing="1">
                {f.tag}
              </text>
              <text x={x - 46} y="371" fontSize="12.5" fill={ink} fontWeight="600" letterSpacing="0.6" className="sld-label">
                {f.title}
              </text>
              <text x={x - 46} y="389" fontSize="10.5" fill={muted} fontFamily="var(--font-sans), sans-serif">
                {f.lines[0]}
              </text>
              <text x={x - 46} y="403" fontSize="10.5" fill={muted} fontFamily="var(--font-sans), sans-serif">
                {f.lines[1]}
              </text>
              <path d={`M${x + 34} 404 l8 0 m-3 -3 l3 3 l-3 3`} stroke={muted} strokeWidth="1.2" fill="none" className="sld-label-arrow sld-hl" />
            </g>
          </FeederLink>
        )
      })}

      {/* ── "Current" running through the conductors ───────────────────── */}
      <g
        aria-hidden
        stroke="hsl(var(--accent))"
        strokeWidth="2"
        fill="none"
        strokeLinecap="round"
        style={{ opacity: 0, animation: "sldFade 600ms 2.6s forwards" }}
      >
        <path d="M300 73 V104 M300 120 V150" className="sld-flow" />
        <path d={`M60 ${BUS_Y} H540`} className="sld-flow" />
        {X.map((x) => (
          <path key={x} d={`M${x} ${BUS_Y} V200 M${x} 238 V330`} className="sld-flow" />
        ))}
      </g>

      {/* Drawing reference, bottom-right */}
      <g className="sld-fade" style={{ animationDelay: "2s" }}>
        <text x="540" y="452" fontSize="9.5" fill={muted} textAnchor="end" letterSpacing="1.2">
          DWG AN-001 · SLD · REV OCT 2026
        </text>
      </g>
    </svg>
  )
}
