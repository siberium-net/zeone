import logging
import os
import sys
import subprocess
from pathlib import Path
from typing import Tuple, List

from git import Repo, InvalidGitRepositoryError, NoSuchPathError, GitCommandError

logger = logging.getLogger(__name__)


class UpdateManager:
    """Safe git-based updater with optional restart."""

    def __init__(self, project_root: str):
        self.root = Path(project_root).resolve()
        try:
            self.repo = Repo(self.root)
        except (InvalidGitRepositoryError, NoSuchPathError):
            self.repo = None
            logger.warning("[UPDATER] Not a git repository: %s", self.root)
    
    def check_update_available(self) -> Tuple[bool, List[str]]:
        """Fetch remote and check if upstream has new commits."""
        if not self.repo:
            return False, []
        try:
            self.repo.git.fetch()
            local = self.repo.git.rev_parse("HEAD")
            try:
                upstream = self.repo.git.rev_parse("@{u}")
            except GitCommandError:
                logger.warning("[UPDATER] No upstream set")
                return False, []
            if local == upstream:
                return False, []
            # Show short log between local..upstream
            log = self.repo.git.log(f"{local}..{upstream}", "--oneline").splitlines()
            return True, log
        except Exception as e:
            logger.warning(f"[UPDATER] check_update_available failed: {e}")
            return False, []
    
    def perform_update(self) -> bool:
        """Pull latest, install deps, run migrations."""
        if not self.repo:
            return False
        try:
            # Stash local changes to avoid loss
            try:
                self.repo.git.stash("save")
            except GitCommandError:
                pass
            self.repo.git.pull()
            # Reinstall requirements
            req = self.root / "requirements.txt"
            if req.exists():
                subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(req)], check=False)
            # Run migrations
            try:
                from core.persistence.migrations import run_migrations
                run_migrations(self.root)
            except Exception as e:
                logger.warning(f"[UPDATER] migrations failed: {e}")
            return True
        except Exception as e:
            logger.error(f"[UPDATER] perform_update failed: {e}")
            return False
    
    def restart_node(self) -> None:
        """Restart current process with same args."""
        python = sys.executable
        os.execv(python, [python] + sys.argv)
