import { Link, useLocation } from "wouter";
import { LayoutDashboard, ScanSearch, FolderKanban, FilePlus, Settings } from "lucide-react";
import { cn } from "@/lib/utils";
import logo from "@/assets/logo.png";

export function Sidebar() {
  const [location] = useLocation();

  const navItems = [
    { href: "/", label: "Обзор (Dashboard)", icon: LayoutDashboard },
    { href: "/recognize", label: "Опознание (Recognize)", icon: ScanSearch },
    { href: "/projects", label: "Проекты (Projects)", icon: FolderKanban },
    { href: "/cards/new", label: "Новая карточка (New Card)", icon: FilePlus },
  ];

  return (
    <div className="w-64 border-r bg-sidebar flex flex-col h-screen fixed left-0 top-0">
      <div className="p-6 border-b flex items-center gap-3">
        <div className="w-8 h-8 rounded-md bg-primary flex items-center justify-center text-primary-foreground font-bold">
          <img src={logo} />
        </div>
        <span className="font-semibold text-lg text-sidebar-foreground">NewtTracker</span>
      </div>
      <nav className="flex-1 p-4 space-y-2">
        {navItems.map((item) => {
          const isActive = location === item.href || (item.href !== "/" && location.startsWith(item.href));
          const Icon = item.icon;
          return (
            <Link key={item.href} href={item.href}>
              <div className={cn(
                "flex items-center gap-3 px-3 py-2.5 rounded-md cursor-pointer transition-colors",
                isActive 
                  ? "bg-sidebar-primary text-sidebar-primary-foreground font-medium" 
                  : "text-sidebar-foreground hover:bg-sidebar-accent hover:text-sidebar-accent-foreground"
              )}>
                <Icon className="w-5 h-5" />
                <span>{item.label}</span>
              </div>
            </Link>
          );
        })}
      </nav>
      <div className="p-4 border-t">
        <div className="flex items-center gap-3 px-3 py-2.5 rounded-md cursor-pointer text-sidebar-foreground hover:bg-sidebar-accent transition-colors">
          <Settings className="w-5 h-5" />
          <span>Настройки</span>
        </div>
      </div>
    </div>
  );
}

export function Layout({ children }: { children: React.ReactNode }) {
  return (
    <div className="min-h-screen bg-background flex">
      <Sidebar />
      <main className="flex-1 ml-64 min-h-screen overflow-y-auto">
        {children}
      </main>
    </div>
  );
}
