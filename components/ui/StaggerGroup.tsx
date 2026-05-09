"use client"

import { useEffect, useRef, useState, Children } from "react"
import type { ReactNode } from "react"

interface StaggerGroupProps {
  children: ReactNode
  /** Delay between each child's animation start (ms). Default 80. */
  staggerDelay?: number
  /** Total animation duration per child (ms). Default 500. */
  duration?: number
  /** Direction of entrance: 'up' | 'down' | 'left' | 'right' | 'scale'. Default 'up'. */
  direction?: "up" | "down" | "left" | "right" | "scale"
  /** Distance for translate transforms (px). Default 12. */
  distance?: number
  /** IntersectionObserver threshold. Default 0.05. */
  threshold?: number
  /** When true (default) animation only plays once. */
  once?: boolean
  className?: string
}

const directions: Record<string, (d: number) => string> = {
  up:    (d) => `translateY(${d}px)`,
  down:  (d) => `translateY(${-d}px)`,
  left:  (d) => `translateX(${d}px)`,
  right: (d) => `translateX(${-d}px)`,
  scale: () => "scale(0.95)",
}

export default function StaggerGroup({
  children,
  staggerDelay = 80,
  duration = 500,
  direction = "up",
  distance = 12,
  threshold = 0.05,
  once = true,
  className,
}: StaggerGroupProps) {
  const ref = useRef<HTMLDivElement>(null)
  const [visible, setVisible] = useState(false)

  useEffect(() => {
    const query = window.matchMedia("(prefers-reduced-motion: reduce)")
    if (query.matches) {
      setVisible(true)
      return
    }

    const el = ref.current
    if (!el) return

    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setVisible(true)
          if (once) observer.unobserve(el)
        }
      },
      { threshold },
    )

    observer.observe(el)
    return () => observer.disconnect()
  }, [threshold, once])

  const translate = directions[direction](distance)

  return (
    <div ref={ref} className={className}>
      {Children.map(children, (child, i) => (
        <div
          style={{
            opacity: visible ? 1 : 0,
            transform: visible ? "none" : translate,
            transition: `opacity ${duration}ms cubic-bezier(0.22,1,0.36,1) ${i * staggerDelay}ms, transform ${duration}ms cubic-bezier(0.22,1,0.36,1) ${i * staggerDelay}ms`,
          }}
        >
          {child}
        </div>
      ))}
    </div>
  )
}
