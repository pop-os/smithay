//! Utilities for manipulating the primary selection
//!
//! The primary selection is an additional protocol modeled after the data device to represent
//! and additional selection (copy/paste), a concept taken from the X11 Server.
//! This primary selection is a shortcut to the common clipboard selection,
//! where text just needs to be selected in order to allow copying it elsewhere
//! The de facto way to perform this action is the middle mouse button, although it is not limited to this one.
//!
//! This module provides the freestanding [`set_primary_focus`] function:
//!   This function sets the data device focus for a given seat; you'd typically call it
//!   whenever the keyboard focus changes, to follow it (for example in the focus hook of your keyboards).
//!
//! The module also provides an additional mechanism allowing your compositor to see and interact with
//! the contents of the primary selection:
//!
//! - the freestanding function [`set_primary_selection`]
//!   allows you to set the contents of the selection for your clients
//! - the `PrimarySelectionHandle` gives you the option to inspect new selections
//!   by overriding [`PrimarySelectionHandler::new_selection].
//!
//! ## Initialization
//!
//! To initialize this implementation, create the [`PrimarySelectionState`], store it inside your `State` struct
//! and implement the [`PrimarySelectionHandler`], as shown in this example:
//!
//! ```
//! # extern crate wayland_server;
//! # #[macro_use] extern crate smithay;
//! use smithay::delegate_primary_selection;
//! use smithay::wayland::primary_selection::{PrimarySelectionState, PrimarySelectionHandler};
//! # use smithay::input::{Seat, SeatHandler, SeatState, pointer::CursorImageStatus};
//! # use smithay::reexports::wayland_server::protocol::wl_surface::WlSurface;
//!
//! # struct State { primary_selection_state: PrimarySelectionState }
//! # let mut display = wayland_server::Display::<State>::new().unwrap();
//! // Create the primary_selection state
//! let primary_selection_state = PrimarySelectionState::new::<State>(
//!     &display.handle(),
//! );
//!
//! // insert the PrimarySelectionState into your state
//! // ..
//!
//! // implement the necessary traits
//! # impl SeatHandler for State {
//! #     type KeyboardFocus = WlSurface;
//! #     type PointerFocus = WlSurface;
//! #     fn seat_state(&mut self) -> &mut SeatState<Self> { unimplemented!() }
//! #     fn focus_changed(&mut self, seat: &Seat<Self>, focused: Option<&WlSurface>) { unimplemented!() }
//! #     fn cursor_image(&mut self, seat: &Seat<Self>, image: CursorImageStatus) { unimplemented!() }
//! # }
//! impl PrimarySelectionHandler for State {
//!     type SelectionUserData = ();
//!     fn primary_selection_state(&self) -> &PrimarySelectionState { &self.primary_selection_state }
//!     // ... override default implementations here to customize handling ...
//! }
//! delegate_primary_selection!(State);
//!
//! // You're now ready to go!
//! ```

use std::{
    cell::{Ref, RefCell},
    os::unix::io::OwnedFd,
};

use tracing::instrument;
use wayland_protocols::wp::primary_selection::zv1::server::{
    zwp_primary_selection_device_manager_v1::ZwpPrimarySelectionDeviceManagerV1 as PrimaryDeviceManager,
    zwp_primary_selection_source_v1::ZwpPrimarySelectionSourceV1 as PrimarySource,
};
use wayland_server::{backend::GlobalId, Client, DisplayHandle, GlobalDispatch};

use crate::input::{Seat, SeatHandler};

mod device;
mod seat_data;
mod source;

pub use device::PrimaryDeviceUserData;
pub use source::{with_source_metadata, PrimarySourceUserData, SourceMetadata};

use seat_data::{SeatData, Selection};

/// Events that are generated by interactions of the clients with the data device
pub trait PrimarySelectionHandler: Sized + SeatHandler {
    /// UserData attached to server-side selections
    type SelectionUserData: Clone + Send + Sync + 'static;

    /// [PrimarySelectionState] getter
    fn primary_selection_state(&self) -> &PrimarySelectionState;

    /// A client has set the selection
    #[allow(unused_variables)]
    fn new_selection(&mut self, source: Option<PrimarySource>, seat: Seat<Self>) {}

    /// A client requested to read the server-set selection
    ///
    /// * `mime_type` - the requested mime type
    /// * `fd` - the fd to write into
    #[allow(unused_variables)]
    fn send_selection(
        &mut self,
        mime_type: String,
        fd: OwnedFd,
        seat: Seat<Self>,
        user_data: &Self::SelectionUserData,
    ) {
    }
}

/// State of data device
#[derive(Debug)]
pub struct PrimarySelectionState {
    manager_global: GlobalId,
}

impl PrimarySelectionState {
    /// Regiseter new [ZwpPrimarySelectionDeviceManagerV1] global
    pub fn new<D>(display: &DisplayHandle) -> Self
    where
        D: GlobalDispatch<PrimaryDeviceManager, ()> + 'static,
        D: PrimarySelectionHandler,
    {
        let manager_global = display.create_global::<D, PrimaryDeviceManager, _>(1, ());

        Self { manager_global }
    }

    /// [ZwpPrimarySelectionDeviceManagerV1] GlobalId getter
    pub fn global(&self) -> GlobalId {
        self.manager_global.clone()
    }
}

/// Set the primary selection focus to a certain client for a given seat
#[instrument(name = "wayland_primary_selection", level = "debug", skip(dh, seat, client), fields(seat = seat.name(), client = ?client.as_ref().map(|c| c.id())))]
pub fn set_primary_focus<D>(dh: &DisplayHandle, seat: &Seat<D>, client: Option<Client>)
where
    D: SeatHandler + PrimarySelectionHandler + 'static,
{
    seat.user_data()
        .insert_if_missing(|| RefCell::new(SeatData::<D::SelectionUserData>::new()));
    let seat_data = seat
        .user_data()
        .get::<RefCell<SeatData<D::SelectionUserData>>>()
        .unwrap();
    seat_data.borrow_mut().set_focus::<D>(dh, client);
}

/// Set a compositor-provided primary selection for this seat
///
/// You need to provide the available mime types for this selection.
///
/// Whenever a client requests to read the selection, your callback will
/// receive a [`PrimarySelectionHandler::send_selection`] event.
#[instrument(name = "wayland_primary_selection", level = "debug", skip(dh, seat, user_data), fields(seat = seat.name()))]
pub fn set_primary_selection<D>(
    dh: &DisplayHandle,
    seat: &Seat<D>,
    mime_types: Vec<String>,
    user_data: D::SelectionUserData,
) where
    D: SeatHandler + PrimarySelectionHandler + 'static,
{
    seat.user_data()
        .insert_if_missing(|| RefCell::new(SeatData::<D::SelectionUserData>::new()));
    let seat_data = seat
        .user_data()
        .get::<RefCell<SeatData<D::SelectionUserData>>>()
        .unwrap();
    seat_data.borrow_mut().set_selection::<D>(
        dh,
        Selection::Compositor {
            metadata: SourceMetadata { mime_types },
            user_data,
        },
    );
}

/// Gets the user_data for the currently active selection, if set by the compositor
#[instrument(name = "wayland_primary_selection", level = "debug", skip_all, fields(seat = seat.name()))]
pub fn current_primary_selection_userdata<D>(seat: &Seat<D>) -> Option<Ref<'_, D::SelectionUserData>>
where
    D: SeatHandler + PrimarySelectionHandler + 'static,
{
    seat.user_data()
        .insert_if_missing(|| RefCell::new(SeatData::<D::SelectionUserData>::new()));
    let seat_data = seat
        .user_data()
        .get::<RefCell<SeatData<D::SelectionUserData>>>()
        .unwrap();
    Ref::filter_map(seat_data.borrow(), |data| match data.get_selection() {
        Selection::Compositor { ref user_data, .. } => Some(user_data),
        _ => None,
    })
    .ok()
}

/// Clear the current selection for this seat
#[instrument(name = "wayland_primary_selection", level = "debug", skip_all, fields(seat = seat.name()))]
pub fn clear_primary_selection<D>(dh: &DisplayHandle, seat: &Seat<D>)
where
    D: SeatHandler + PrimarySelectionHandler + 'static,
{
    seat.user_data()
        .insert_if_missing(|| RefCell::new(SeatData::<D::SelectionUserData>::new()));
    let seat_data = seat
        .user_data()
        .get::<RefCell<SeatData<D::SelectionUserData>>>()
        .unwrap();
    seat_data.borrow_mut().set_selection::<D>(dh, Selection::Empty);
}

mod handlers {
    use std::cell::RefCell;

    use tracing::error;
    use wayland_protocols::wp::primary_selection::zv1::server::{
        zwp_primary_selection_device_manager_v1::{
            self as primary_device_manager, ZwpPrimarySelectionDeviceManagerV1 as PrimaryDeviceManager,
        },
        zwp_primary_selection_device_v1::ZwpPrimarySelectionDeviceV1 as PrimaryDevice,
        zwp_primary_selection_source_v1::ZwpPrimarySelectionSourceV1 as PrimarySource,
    };
    use wayland_server::{Dispatch, DisplayHandle, GlobalDispatch};

    use crate::input::{Seat, SeatHandler};

    use super::{device::PrimaryDeviceUserData, seat_data::SeatData, source::PrimarySourceUserData};
    use super::{PrimarySelectionHandler, PrimarySelectionState};

    impl<D> GlobalDispatch<PrimaryDeviceManager, (), D> for PrimarySelectionState
    where
        D: GlobalDispatch<PrimaryDeviceManager, ()>,
        D: Dispatch<PrimaryDeviceManager, ()>,
        D: Dispatch<PrimarySource, PrimarySourceUserData>,
        D: Dispatch<PrimaryDevice, PrimaryDeviceUserData>,
        D: PrimarySelectionHandler,
        D: 'static,
    {
        fn bind(
            _state: &mut D,
            _handle: &DisplayHandle,
            _client: &wayland_server::Client,
            resource: wayland_server::New<PrimaryDeviceManager>,
            _global_data: &(),
            data_init: &mut wayland_server::DataInit<'_, D>,
        ) {
            data_init.init(resource, ());
        }
    }

    impl<D> Dispatch<PrimaryDeviceManager, (), D> for PrimarySelectionState
    where
        D: Dispatch<PrimaryDeviceManager, ()>,
        D: Dispatch<PrimarySource, PrimarySourceUserData>,
        D: Dispatch<PrimaryDevice, PrimaryDeviceUserData>,
        D: PrimarySelectionHandler,
        D: SeatHandler,
        D: 'static,
    {
        fn request(
            _state: &mut D,
            client: &wayland_server::Client,
            _resource: &PrimaryDeviceManager,
            request: primary_device_manager::Request,
            _data: &(),
            _dhandle: &DisplayHandle,
            data_init: &mut wayland_server::DataInit<'_, D>,
        ) {
            match request {
                primary_device_manager::Request::CreateSource { id } => {
                    data_init.init(id, PrimarySourceUserData::new());
                }
                primary_device_manager::Request::GetDevice { id, seat: wl_seat } => {
                    match Seat::<D>::from_resource(&wl_seat) {
                        Some(seat) => {
                            seat.user_data()
                                .insert_if_missing(|| RefCell::new(SeatData::<D::SelectionUserData>::new()));

                            let device = data_init.init(id, PrimaryDeviceUserData { wl_seat });

                            let seat_data = seat
                                .user_data()
                                .get::<RefCell<SeatData<D::SelectionUserData>>>()
                                .unwrap();
                            seat_data.borrow_mut().add_device(device);
                        }
                        None => {
                            error!(
                                primary_selection_device = ?id,
                                client = ?client,
                                "Unmanaged seat given to a primary selection device."
                            );
                        }
                    }
                }
                primary_device_manager::Request::Destroy => {}
                _ => unreachable!(),
            }
        }
    }
}

#[allow(missing_docs)] // TODO
#[macro_export]
macro_rules! delegate_primary_selection {
    ($(@<$( $lt:tt $( : $clt:tt $(+ $dlt:tt )* )? ),+>)? $ty: ty) => {
        $crate::reexports::wayland_server::delegate_global_dispatch!($(@< $( $lt $( : $clt $(+ $dlt )* )? ),+ >)? $ty: [
            $crate::reexports::wayland_protocols::wp::primary_selection::zv1::server::zwp_primary_selection_device_manager_v1::ZwpPrimarySelectionDeviceManagerV1: ()
        ] => $crate::wayland::primary_selection::PrimarySelectionState);

        $crate::reexports::wayland_server::delegate_dispatch!($(@< $( $lt $( : $clt $(+ $dlt )* )? ),+ >)? $ty: [
            $crate::reexports::wayland_protocols::wp::primary_selection::zv1::server::zwp_primary_selection_device_manager_v1::ZwpPrimarySelectionDeviceManagerV1: ()
        ] => $crate::wayland::primary_selection::PrimarySelectionState);
        $crate::reexports::wayland_server::delegate_dispatch!($(@< $( $lt $( : $clt $(+ $dlt )* )? ),+ >)? $ty: [
            $crate::reexports::wayland_protocols::wp::primary_selection::zv1::server::zwp_primary_selection_device_v1::ZwpPrimarySelectionDeviceV1: $crate::wayland::primary_selection::PrimaryDeviceUserData
        ] => $crate::wayland::primary_selection::PrimarySelectionState);
        $crate::reexports::wayland_server::delegate_dispatch!($(@< $( $lt $( : $clt $(+ $dlt )* )? ),+ >)? $ty: [
            $crate::reexports::wayland_protocols::wp::primary_selection::zv1::server::zwp_primary_selection_source_v1::ZwpPrimarySelectionSourceV1: $crate::wayland::primary_selection::PrimarySourceUserData
        ] => $crate::wayland::primary_selection::PrimarySelectionState);
    };
}
