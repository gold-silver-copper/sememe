//! The discourse context: what has been said so far in a document.
//!
//! Reading is adding to the context. It holds the conversation's speakers,
//! the entities mentioned so far (most recent first) and the meanings of
//! earlier fragments, so later meanings can refer back to them or restate
//! them cheaply. It has no fixed size; only the search looks back a bounded
//! distance.

use std::collections::VecDeque;
use std::sync::Arc;

use crate::cost::{FACT_LOOKBACK, REF_LOOKBACK, SPEAKER_LOOKBACK};
use crate::program::SPEAKER;
use crate::term::Term;
use crate::text::TagKind;

/// An entity in the context.
#[derive(Clone, Debug, PartialEq)]
pub struct EntityRec {
    /// Context-local ID.
    pub id: u32,
    /// Head key of its concept (see `cost::head_key`).
    pub concept: u64,
    /// Its proper name, if any.
    pub name: Option<Arc<str>>,
}

/// A remembered fragment meaning.
#[derive(Clone, Debug)]
pub struct FactRec {
    /// `cost::meaning_key` of the meaning.
    pub key: u64,
    /// The meaning itself.
    pub meaning: Term,
}

/// The context of one document.
#[derive(Clone, Debug)]
pub struct Context {
    /// Entities, most recent first.
    pub entities: VecDeque<EntityRec>,
    /// Every entity ever introduced, by ID.
    pub all: Vec<EntityRec>,
    /// The current speaker entity.
    pub speaker: u32,
    /// The current addressee entity.
    pub addressee: u32,
    /// Recent speaker names, most recent first.
    pub speakers: VecDeque<String>,
    /// Speaker entity by name.
    pub speaker_ids: Vec<(String, u32)>,
    /// Earlier fragment meanings, most recent first.
    pub facts: VecDeque<FactRec>,
    /// The previous line's tag kind.
    pub prev_tag: TagKind,
    dash: [u32; 2],
    dash_turn: usize,
}

impl Default for Context {
    fn default() -> Self {
        Context::new()
    }
}

impl Context {
    /// A fresh context: a narrator speaking to a reader.
    pub fn new() -> Context {
        let mut c = Context {
            entities: VecDeque::new(),
            all: Vec::new(),
            speaker: 0,
            addressee: 0,
            speakers: VecDeque::new(),
            speaker_ids: Vec::new(),
            facts: VecDeque::new(),
            prev_tag: TagKind::None,
            dash: [0, 0],
            dash_turn: 0,
        };
        c.speaker = c.new_entity(u64::from(SPEAKER), None);
        c.addressee = c.new_entity(u64::from(SPEAKER), None);
        let a = c.new_entity(u64::from(SPEAKER), None);
        let b = c.new_entity(u64::from(SPEAKER), None);
        c.dash = [a, b];
        c
    }

    /// Introduce a new entity (not yet in the recency list).
    pub fn new_entity(&mut self, concept: u64, name: Option<Arc<str>>) -> u32 {
        let id = u32::try_from(self.all.len()).unwrap_or(u32::MAX);
        self.all.push(EntityRec { id, concept, name });
        id
    }

    /// The entity with this ID.
    pub fn entity(&self, id: u32) -> Option<&EntityRec> {
        self.all.get(id as usize)
    }

    /// Move an entity to the front of the recency list.
    pub fn mention(&mut self, id: u32) {
        if let Some(pos) = self.entities.iter().position(|e| e.id == id) {
            self.entities.remove(pos);
        }
        if let Some(e) = self.all.get(id as usize).cloned() {
            self.entities.push_front(e);
        }
        while self.entities.len() > 4 * REF_LOOKBACK {
            self.entities.pop_back();
        }
    }

    /// The reference rank of an entity: 0 speaker, 1 addressee, then
    /// 2 + position in the recency list (if within the lookback).
    pub fn rank_of(&self, id: u32) -> Option<usize> {
        if id == self.speaker {
            return Some(0);
        }
        if id == self.addressee {
            return Some(1);
        }
        self.recent().position(|e| e.id == id).map(|p| p + 2)
    }

    /// Entities within the reference lookback, most recent first,
    /// excluding the speaker and addressee.
    pub fn recent(&self) -> impl Iterator<Item = &EntityRec> {
        let (s, a) = (self.speaker, self.addressee);
        self.entities
            .iter()
            .filter(move |e| e.id != s && e.id != a)
            .take(REF_LOOKBACK)
    }

    /// A speaker turn. Returns the speaker's rank among recent speakers
    /// (`None` for a new speaker).
    pub fn turn(&mut self, tag: TagKind, name: Option<&str>) -> Option<usize> {
        let mut rank = None;
        match (tag, name) {
            (TagKind::Childes | TagKind::Swb, Some(name)) => {
                rank = self
                    .speakers
                    .iter()
                    .position(|s| s == name)
                    .filter(|r| *r < SPEAKER_LOOKBACK);
                if let Some(pos) = self.speakers.iter().position(|s| s == name) {
                    self.speakers.remove(pos);
                }
                self.speakers.push_front(name.to_string());
                while self.speakers.len() > SPEAKER_LOOKBACK {
                    self.speakers.pop_back();
                }
                let id = match self.speaker_ids.iter().find(|(n, _)| n == name) {
                    Some((_, id)) => *id,
                    None => {
                        let id = self.new_entity(
                            u64::from(SPEAKER),
                            Some(Arc::from(name.to_lowercase().as_str())),
                        );
                        self.speaker_ids.push((name.to_string(), id));
                        id
                    }
                };
                if id != self.speaker {
                    self.addressee = self.speaker;
                    self.speaker = id;
                }
            }
            (TagKind::Dash, _) => {
                self.dash_turn = 1 - self.dash_turn.min(1);
                let id = self.dash.get(self.dash_turn).copied().unwrap_or(0);
                let other = self.dash.get(1 - self.dash_turn).copied().unwrap_or(0);
                self.speaker = id;
                self.addressee = other;
            }
            _ => {}
        }
        self.prev_tag = tag;
        rank
    }

    /// Remember a fragment meaning.
    pub fn remember(&mut self, key: u64, meaning: Term) {
        self.facts.push_front(FactRec { key, meaning });
        while self.facts.len() > FACT_LOOKBACK {
            self.facts.pop_back();
        }
    }
}
